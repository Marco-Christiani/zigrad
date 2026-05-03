//! Zig API for PJRT
//!
//! These are thin wrappers calling through to api.zig.
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c = @import("c.zig").c;

const log = std.log.scoped(.@"zg/pjrt/types");

// TODO: now that we have proto bindings we can go deeper, not justified now since
//  we currently do not need much in the way of compilation options.

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

fn make_named_value_int64(name: []const u8, value: i64) c.PJRT_NamedValue {
    var out: c.PJRT_NamedValue = std.mem.zeroes(c.PJRT_NamedValue);
    out.struct_size = api_mod.pjrt_struct_size(c.PJRT_NamedValue);
    out.extension_start = null;
    out.name = name.ptr;
    out.name_size = name.len;
    out.type = c.PJRT_NamedValue_kInt64;
    out.unnamed_0.int64_value = value;
    out.value_size = 1;
    return out;
}

pub const Client = struct {
    api: *Api,
    pjrt_client: *c.PJRT_Client,

    pub fn create(api: *Api) !Client {
        return create_with_options(api, null);
    }

    pub fn create_with_options(api: *Api, create_options: ?[]const c.PJRT_NamedValue) !Client {
        var args = api_mod.init_args(c.PJRT_Client_Create_Args);
        const empty_opts: [0]c.PJRT_NamedValue = .{};
        args.create_options = if (create_options) |opts| opts.ptr else @ptrCast(&empty_opts);
        args.num_options = if (create_options) |opts| opts.len else 0;

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

    pub fn create_cpu_with_device_count(api: *Api, cpu_device_count: usize) !Client {
        const option = make_named_value_int64("cpu_device_count", @intCast(cpu_device_count));
        const options = [_]c.PJRT_NamedValue{option};
        return create_with_options(api, options[0..]);
    }

    pub fn deinit(self: *Client) void {
        var args = api_mod.init_args(c.PJRT_Client_Destroy_Args);
        args.client = self.pjrt_client;
        self.api.call("PJRT_Client_Destroy", &args) catch {};
    }

    /// Returned slice is owned by caller.
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

        // create PJRT_Program
        var program = api_mod.init_args(c.PJRT_Program);
        program.code = @constCast(bytecode.ptr);
        program.code_size = bytecode.len;

        // TODO: audit and document this
        const format_str = switch (format) {
            .mlir_text => "mlir",
            .mlir_bytecode => "mlir",
            .stablehlo_portable => "mlir",
        };
        program.format = format_str.ptr;
        program.format_size = format_str.len;

        // build args and compile
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

    pub fn get_topology_description(self: *Client) !TopologyDescription {
        var args = api_mod.init_args(c.PJRT_Client_TopologyDescription_Args);
        args.client = self.pjrt_client;
        args.topology = null;

        try self.api.call("PJRT_Client_TopologyDescription", &args);

        const topo_ptr = args.topology orelse return error.PjrtReturnedNullTopology;
        return TopologyDescription{
            .api = self.api,
            .pjrt_topology = topo_ptr,
            .owned = false,
        };
    }

    pub fn compile_aot(
        self: *Client,
        topology: *const TopologyDescription,
        format: ProgramFormat,
        bytecode: []const u8,
        options: ?[]const u8,
    ) !Executable {
        // create PJRT_Program
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

        // build args and compile
        var args = api_mod.init_args(c.PJRT_Compile_Args);
        args.topology = topology.pjrt_topology;
        args.program = &program;
        args.compile_options = if (options) |opts| opts.ptr else &minimal_compile_opts;
        args.compile_options_size = if (options) |opts| opts.len else minimal_compile_opts.len;
        args.client = self.pjrt_client;
        args.executable = null;

        try self.api.call("PJRT_Compile", &args);

        const executable_ptr = args.executable orelse return error.PjrtReturnedNullExecutable;
        return Executable{
            .pjrt_executable = executable_ptr,
        };
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
            .pjrt_buffer = buffer_ptr,
        };
    }
};

/// Used in the AOT path.
pub const TopologyDescription = struct {
    api: *Api,
    pjrt_topology: *c.PJRT_TopologyDescription,
    owned: bool,

    pub fn deinit(self: *TopologyDescription) void {
        if (!self.owned) return;
        var args = api_mod.init_args(c.PJRT_TopologyDescription_Destroy_Args);
        args.topology = self.pjrt_topology;
        self.api.call("PJRT_TopologyDescription_Destroy", &args) catch {};
    }
};

pub const Device = struct {
    pjrt_device: *c.PJRT_Device,

    pub const MemoryStats = struct {
        bytes_in_use: i64,
        peak_bytes_in_use: ?i64,
        num_allocs: ?i64,
        largest_alloc_size: ?i64,
        bytes_limit: ?i64,
        bytes_reserved: ?i64,
        peak_bytes_reserved: ?i64,
        bytes_reservable_limit: ?i64,
        largest_free_block_bytes: ?i64,
        pool_bytes: ?i64,
        peak_pool_bytes: ?i64,
    };

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

    pub fn get_memory_stats(self: *const Device, api: *Api) !MemoryStats {
        var args = api_mod.init_args(c.PJRT_Device_MemoryStats_Args);
        args.device = self.pjrt_device;

        try api.call("PJRT_Device_MemoryStats", &args);

        return .{
            .bytes_in_use = args.bytes_in_use,
            .peak_bytes_in_use = if (args.peak_bytes_in_use_is_set) args.peak_bytes_in_use else null,
            .num_allocs = if (args.num_allocs_is_set) args.num_allocs else null,
            .largest_alloc_size = if (args.largest_alloc_size_is_set) args.largest_alloc_size else null,
            .bytes_limit = if (args.bytes_limit_is_set) args.bytes_limit else null,
            .bytes_reserved = if (args.bytes_reserved_is_set) args.bytes_reserved else null,
            .peak_bytes_reserved = if (args.peak_bytes_reserved_is_set) args.peak_bytes_reserved else null,
            .bytes_reservable_limit = if (args.bytes_reservable_limit_is_set) args.bytes_reservable_limit else null,
            .largest_free_block_bytes = if (args.largest_free_block_bytes_is_set) args.largest_free_block_bytes else null,
            .pool_bytes = if (args.pool_bytes_is_set) args.pool_bytes else null,
            .peak_pool_bytes = if (args.peak_pool_bytes_is_set) args.peak_pool_bytes else null,
        };
    }
};

pub const Executable = struct {
    pjrt_executable: *c.PJRT_Executable,

    pub fn deinit(self: *Executable, api: *Api) void {
        var args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
        args.executable = self.pjrt_executable;
        api.call("PJRT_Executable_Destroy", &args) catch {};
    }

    pub fn serialize(self: *Executable, api: *Api, allocator: std.mem.Allocator) ![]u8 {
        var args = api_mod.init_args(c.PJRT_Executable_Serialize_Args);
        args.executable = self.pjrt_executable;

        try api.call("PJRT_Executable_Serialize", &args);

        const serialized = args.serialized_executable orelse return error.PjrtReturnedNullSerializedExecutable;
        const deleter = args.serialized_executable_deleter orelse return error.PjrtReturnedNullSerializedExecutableDeleter;
        defer deleter(serialized);

        if (args.serialized_bytes == null and args.serialized_bytes_size != 0) {
            return error.PjrtReturnedNullSerializedBytes;
        }

        const bytes = args.serialized_bytes[0..args.serialized_bytes_size];
        return allocator.dupe(u8, bytes);
    }
};

pub const LoadedExecutable = struct {
    pjrt_executable: *c.PJRT_LoadedExecutable,
    num_outputs: usize,

    pub const CompiledMemoryStats = struct {
        generated_code_size_in_bytes: i64,
        argument_size_in_bytes: i64,
        output_size_in_bytes: i64,
        alias_size_in_bytes: i64,
        temp_size_in_bytes: i64,
        host_generated_code_size_in_bytes: i64,
        host_argument_size_in_bytes: i64,
        host_output_size_in_bytes: i64,
        host_alias_size_in_bytes: i64,
        host_temp_size_in_bytes: i64,
        peak_memory_in_bytes: i64,
    };

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
            .pjrt_executable = pjrt_executable,
            .num_outputs = num_outputs,
        };
    }

    pub fn deinit(self: *LoadedExecutable, api: *Api) void {
        var args = api_mod.init_args(c.PJRT_LoadedExecutable_Destroy_Args);
        args.executable = self.pjrt_executable;
        api.call("PJRT_LoadedExecutable_Destroy", &args) catch {};
    }

    pub fn serialize(self: *LoadedExecutable, api: *Api, allocator: std.mem.Allocator) ![]u8 {
        // Get the underlying PJRT_Executable to serialize
        var get_exec_args = api_mod.init_args(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = self.pjrt_executable;
        get_exec_args.executable = null;
        try api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);

        const pjrt_executable = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_executable;
            api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        var args = api_mod.init_args(c.PJRT_Executable_Serialize_Args);
        args.executable = pjrt_executable;

        try api.call("PJRT_Executable_Serialize", &args);

        const serialized = args.serialized_executable orelse return error.PjrtReturnedNullSerializedExecutable;
        const deleter = args.serialized_executable_deleter orelse return error.PjrtReturnedNullSerializedExecutableDeleter;
        defer deleter(serialized);

        if (args.serialized_bytes == null and args.serialized_bytes_size != 0) {
            return error.PjrtReturnedNullSerializedBytes;
        }

        const bytes = args.serialized_bytes[0..args.serialized_bytes_size];
        return allocator.dupe(u8, bytes);
    }

    pub fn get_compiled_memory_stats(self: *LoadedExecutable, api: *Api) !CompiledMemoryStats {
        var get_exec_args = api_mod.init_args(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = self.pjrt_executable;
        get_exec_args.executable = null;
        try api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);

        const pjrt_exec = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_exec;
            api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        var args = api_mod.init_args(c.PJRT_Executable_GetCompiledMemoryStats_Args);
        args.executable = pjrt_exec;
        try api.call("PJRT_Executable_GetCompiledMemoryStats", &args);

        return .{
            .generated_code_size_in_bytes = args.generated_code_size_in_bytes,
            .argument_size_in_bytes = args.argument_size_in_bytes,
            .output_size_in_bytes = args.output_size_in_bytes,
            .alias_size_in_bytes = args.alias_size_in_bytes,
            .temp_size_in_bytes = args.temp_size_in_bytes,
            .host_generated_code_size_in_bytes = args.host_generated_code_size_in_bytes,
            .host_argument_size_in_bytes = args.host_argument_size_in_bytes,
            .host_output_size_in_bytes = args.host_output_size_in_bytes,
            .host_alias_size_in_bytes = args.host_alias_size_in_bytes,
            .host_temp_size_in_bytes = args.host_temp_size_in_bytes,
            .peak_memory_in_bytes = args.peak_memory_in_bytes,
        };
    }

    pub const OptimizedProgram = struct {
        code: []u8,
        format: []const u8,

        pub fn deinit(self: *OptimizedProgram, allocator: std.mem.Allocator) void {
            allocator.free(self.code);
            allocator.free(self.format);
        }
    };

    /// Retrieve the optimized program (e.g. HLO) from the backend after compilation.
    ///
    /// Uses the two-call PJRT pattern: first call queries size, second fills buffer.
    /// Returns null if the plugin does not implement this API.
    /// Format is backend-dependent (XLA: serialized HloModuleProtoWithConfig).
    pub fn get_optimized_program(self: *LoadedExecutable, api: *Api, allocator: std.mem.Allocator) !?OptimizedProgram {
        // Get underlying PJRT_Executable
        var get_exec_args = api_mod.init_args(c.PJRT_LoadedExecutable_GetExecutable_Args);
        get_exec_args.loaded_executable = self.pjrt_executable;
        get_exec_args.executable = null;
        try api.call("PJRT_LoadedExecutable_GetExecutable", &get_exec_args);

        const pjrt_exec = get_exec_args.executable orelse return error.PjrtReturnedNullExecutable;
        defer {
            var destroy_args = api_mod.init_args(c.PJRT_Executable_Destroy_Args);
            destroy_args.executable = pjrt_exec;
            api.call("PJRT_Executable_Destroy", &destroy_args) catch {};
        }

        // First call: query size (code = null)
        var program = api_mod.init_args(c.PJRT_Program);
        program.code = null;
        program.code_size = 0;
        program.format = null;
        program.format_size = 0;

        var args = api_mod.init_args(c.PJRT_Executable_OptimizedProgram_Args);
        args.executable = pjrt_exec;
        args.program = &program;

        api.call("PJRT_Executable_OptimizedProgram", &args) catch |err| {
            if (err == error.FunctionNotAvailable) return null;
            return err;
        };

        if (program.code_size == 0) return null;

        // Second call: fill buffer
        const code_buf = try allocator.alloc(u8, program.code_size);
        errdefer allocator.free(code_buf);
        program.code = @ptrCast(code_buf.ptr);

        api.call("PJRT_Executable_OptimizedProgram", &args) catch |err| {
            if (err == error.FunctionNotAvailable) return null;
            return err;
        };

        const format = if (program.format != null and program.format_size > 0)
            try allocator.dupe(u8, program.format[0..program.format_size])
        else
            try allocator.dupe(u8, "unknown");

        return .{
            .code = code_buf,
            .format = format,
        };
    }

    pub fn execute(self: *LoadedExecutable, api: *Api, allocator: std.mem.Allocator, inputs: []const Buffer) !ExecuteResult {
        return self.execute_with_context(api, allocator, inputs, null);
    }

    pub fn execute_with_context(
        self: *LoadedExecutable,
        api: *Api,
        allocator: std.mem.Allocator,
        inputs: []const Buffer,
        execute_context: ?*c.PJRT_ExecuteContext,
    ) !ExecuteResult {
        // TODO: Per-stage timing was driven by std.time.Timer. That API is gone
        //  in 0.16 and execute() does not have an `io` handle. Tracing now
        //  emits aggregate logs without per-stage breakdown.
        const trace = api.trace_execute;

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
        execute_opts.context = execute_context;

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

        try api.call("PJRT_LoadedExecutable_Execute", &args);

        const outputs = try allocator.alloc(Buffer, num_outputs);
        for (outputs, 0..) |*buf, i| {
            buf.* = Buffer{
                .pjrt_buffer = output_ptrs[i] orelse return error.PjrtReturnedNullOutputBuffer,
            };
        }

        const event = if (device_events[0]) |ev| Event{ .pjrt_event = ev } else null;
        if (trace) {
            log.info(
                "pjrt execute: inputs={d} outputs={d}",
                .{ inputs.len, num_outputs },
            );
        }
        return .{ .outputs = outputs, .device_complete_event = event };
    }

    pub fn execute_into(
        self: *LoadedExecutable,
        api: *Api,
        input_ptrs: []const *c.PJRT_Buffer,
        output_ptrs: []*c.PJRT_Buffer,
    ) !?Event {
        return self.execute_into_opts_with_context(api, input_ptrs, output_ptrs, null, null);
    }

    pub fn execute_into_opts(
        self: *LoadedExecutable,
        api: *Api,
        input_ptrs: []const *c.PJRT_Buffer,
        output_ptrs: []*c.PJRT_Buffer,
        non_donatable_input_indices: ?[]const i64,
    ) !?Event {
        return self.execute_into_opts_with_context(api, input_ptrs, output_ptrs, non_donatable_input_indices, null);
    }

    /// Execute and write output buffer pointers into caller-provided slots.
    ///
    /// ## C binding nullability
    ///
    /// The C API declares `output_lists` as `PJRT_Buffer** const*`, non-nullable on all levels.
    /// PJRT always writes valid buffer pointers into the caller's array.
    ///
    /// However, Zig's translate-c generates the field as `[*c]const [*c]?*PJRT_Buffer`.
    /// This happens because:
    ///
    ///  1. `PJRT_Buffer` is `opaque {}` (C struct with hidden layout).
    ///  2. `[*c]T` ("C pointer") is inherently nullable -- it allows address 0 -- and is a
    ///      compromise type for auto-generated code.
    ///  3. translate-c cannot distinguish nullable from non-nullable C pointers, so all
    ///      `PJRT_Buffer*` become `?*PJRT_Buffer`.
    ///
    /// The recommended practice is to replace `[*c]T` with proper Zig pointer types (`*T`, `[*]T`)
    ///  in wrapper code. We do exactly that, but instead of editing the generated code this function
    ///  takes `[]*c.PJRT_Buffer` (non-nullable slice) and `@ptrCast`s to the generated `[*c]?*` type
    ///  at the FFI boundary. This cast is sound because for bare pointer types, `*T` and `?*T` have
    ///  identical layout (both pointer-sized, null sentinel).
    /// This would NOT hold for struct-wrapped pointers: `?struct{*T}` is 16 bytes (separate bool tag).
    /// Thus, this must be kept in sync with `Backend.Buffer`.
    pub fn execute_into_opts_with_context(
        self: *LoadedExecutable,
        api: *Api,
        input_ptrs: []const *c.PJRT_Buffer,
        output_ptrs: []*c.PJRT_Buffer,
        non_donatable_input_indices: ?[]const i64,
        execute_context: ?*c.PJRT_ExecuteContext,
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
        execute_opts.context = execute_context;

        var device_events = [_]?*c.PJRT_Event{null};

        var args = api_mod.init_args(c.PJRT_LoadedExecutable_Execute_Args);
        args.executable = self.pjrt_executable;
        args.options = &execute_opts;
        // Cast non-nullable zig pointers to the generated [*c]?* types.
        // See doc comment above for why this is sound.
        args.argument_lists = @ptrCast(&input_lists);
        args.num_devices = 1;
        args.num_args = input_ptrs.len;
        args.output_lists = @ptrCast(&output_lists);
        args.device_complete_events = @ptrCast(&device_events);
        args.execute_device = null;

        try api.call("PJRT_LoadedExecutable_Execute", &args);

        return if (device_events[0]) |ev| Event{ .pjrt_event = ev } else null;
    }
};

pub const ExecuteResult = struct {
    outputs: []Buffer,
    device_complete_event: ?Event,
};

pub const Buffer = struct {
    pjrt_buffer: *c.PJRT_Buffer,

    pub fn deinit(self: *Buffer, api: *Api) void {
        var args = api_mod.init_args(c.PJRT_Buffer_Destroy_Args);
        args.buffer = self.pjrt_buffer;
        api.call("PJRT_Buffer_Destroy", &args) catch {};
    }

    // Returned slice is owned by caller.
    pub fn get_dimensions(self: *const Buffer, api: *Api, allocator: std.mem.Allocator) ![]usize {
        var args = api_mod.init_args(c.PJRT_Buffer_Dimensions_Args);
        args.buffer = self.pjrt_buffer;
        args.dims = null;
        args.num_dims = 0;

        try api.call("PJRT_Buffer_Dimensions", &args);

        const num_dims = args.num_dims;
        const dims_i64 = args.dims orelse return error.PjrtReturnedNullDimensions;

        // Convert from i64 slice to usize slice
        const dims = try allocator.alloc(usize, num_dims);
        for (0..num_dims) |i| {
            dims[i] = @intCast(dims_i64[i]);
        }

        return dims;
    }

    /// Returns the buffer's dimensions as a borrowed slice.
    ///
    /// The returned slice is not allocated and has the lifetime of the buffer.
    pub fn get_dimensions_borrowed(self: *const Buffer, api: *Api) ![]const i64 {
        var args = api_mod.init_args(c.PJRT_Buffer_Dimensions_Args);
        args.buffer = self.pjrt_buffer;
        args.dims = null;
        args.num_dims = 0;
        try api.call("PJRT_Buffer_Dimensions", &args);
        const dims = args.dims orelse return error.PjrtReturnedNullDimensions;
        return dims[0..args.num_dims];
    }

    /// Returns the element type of the buffer.
    pub fn get_element_type(self: *const Buffer, api: *Api) !BufferType {
        var args = api_mod.init_args(c.PJRT_Buffer_ElementType_Args);
        args.buffer = self.pjrt_buffer;
        try api.call("PJRT_Buffer_ElementType", &args);
        return BufferType.from_c_enum(args.type);
    }

    /// Returns the device that owns this buffer.
    pub fn get_device(self: *const Buffer, api: *Api) !Device {
        var args = api_mod.init_args(c.PJRT_Buffer_Device_Args);
        args.buffer = self.pjrt_buffer;
        args.device = null;
        try api.call("PJRT_Buffer_Device", &args);
        const device_ptr = args.device orelse return error.PjrtReturnedNullDevice;
        return Device{ .pjrt_device = device_ptr };
    }

    pub fn to_host(self: *Buffer, api: *Api, dst: []u8) !Event {
        var args = api_mod.init_args(c.PJRT_Buffer_ToHostBuffer_Args);

        args.src = self.pjrt_buffer;
        args.host_layout = null;
        args.dst = dst.ptr;
        args.dst_size = dst.len;
        args.event = null;

        try api.call("PJRT_Buffer_ToHostBuffer", &args);

        const event_ptr = args.event orelse return error.PjrtReturnedNullEvent;
        return Event{
            .pjrt_event = event_ptr,
        };
    }

    pub fn ready_event(self: *Buffer, api: *Api) !Event {
        var args = api_mod.init_args(c.PJRT_Buffer_ReadyEvent_Args);

        args.buffer = self.pjrt_buffer;
        args.event = null;

        try api.call("PJRT_Buffer_ReadyEvent", &args);

        const event_ptr = args.event orelse return error.PjrtReturnedNullEvent;
        return Event{
            .pjrt_event = event_ptr,
        };
    }

    pub fn is_on_cpu(self: *const Buffer, api: *Api) !bool {
        var args = api_mod.init_args(c.PJRT_Buffer_IsOnCpu_Args);
        args.buffer = self.pjrt_buffer;
        args.is_on_cpu = false;
        try api.call("PJRT_Buffer_IsOnCpu", &args);
        return args.is_on_cpu;
    }

    pub fn unsafe_pointer(self: *const Buffer, api: *Api) !usize {
        var args = api_mod.init_args(c.PJRT_Buffer_UnsafePointer_Args);
        args.buffer = self.pjrt_buffer;
        args.buffer_pointer = 0;
        try api.call("PJRT_Buffer_UnsafePointer", &args);
        return args.buffer_pointer;
    }
};

/// Raw C pointer type for zero-copy buffer operations.
/// Use with execute_into for hot paths where Buffer wrapper overhead matters.
pub const RawBuffer = *c.PJRT_Buffer;

pub const Event = struct {
    pjrt_event: *c.PJRT_Event,

    pub fn deinit(self: *Event, api: *Api) void {
        var args = api_mod.init_args(c.PJRT_Event_Destroy_Args);
        args.event = self.pjrt_event;
        api.call("PJRT_Event_Destroy", &args) catch {};
    }

    pub fn await_(self: *Event, api: *Api) !void {
        var args = api_mod.init_args(c.PJRT_Event_Await_Args);
        args.event = self.pjrt_event;
        try api.call("PJRT_Event_Await", &args);
    }

    pub fn is_ready(self: *Event, api: *Api) !bool {
        var args = api_mod.init_args(c.PJRT_Event_IsReady_Args);
        args.event = self.pjrt_event;
        try api.call("PJRT_Event_IsReady", &args);
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
            // TODO: Treating this as bytecode for now, should revisit this
            .stablehlo_portable => c.PJRT_Program_Format_MLIR_BYTECODE,
        };
    }
};

pub const BufferType = enum {
    f16,
    bf16,
    f32,
    f64,
    i8,
    u8,
    i32,
    i64,
    u32,
    u64,

    pub fn from_c_enum(t: c.PJRT_Buffer_Type) !BufferType {
        return switch (t) {
            c.PJRT_Buffer_Type_F16 => .f16,
            c.PJRT_Buffer_Type_BF16 => .bf16,
            c.PJRT_Buffer_Type_F32 => .f32,
            c.PJRT_Buffer_Type_F64 => .f64,
            c.PJRT_Buffer_Type_S8 => .i8,
            c.PJRT_Buffer_Type_U8 => .u8,
            c.PJRT_Buffer_Type_S32 => .i32,
            c.PJRT_Buffer_Type_S64 => .i64,
            c.PJRT_Buffer_Type_U32 => .u32,
            c.PJRT_Buffer_Type_U64 => .u64,
            else => error.UnsupportedBufferType,
        };
    }

    pub fn to_c_enum(self: BufferType) c.PJRT_Buffer_Type {
        return switch (self) {
            .f16 => c.PJRT_Buffer_Type_F16,
            .bf16 => c.PJRT_Buffer_Type_BF16,
            .f32 => c.PJRT_Buffer_Type_F32,
            .f64 => c.PJRT_Buffer_Type_F64,
            .i8 => c.PJRT_Buffer_Type_S8,
            .u8 => c.PJRT_Buffer_Type_U8,
            .i32 => c.PJRT_Buffer_Type_S32,
            .i64 => c.PJRT_Buffer_Type_S64,
            .u32 => c.PJRT_Buffer_Type_U32,
            .u64 => c.PJRT_Buffer_Type_U64,
        };
    }

    pub fn size_in_bytes(self: BufferType) usize {
        return switch (self) {
            .f16, .bf16 => 2,
            .i8, .u8 => 1,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
        };
    }
};
