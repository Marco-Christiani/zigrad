//! PJRT C-API Bindings
//!
//! Low-level wrapper around PJRT C-API with some basic necessities:
//! - Reflection-based call dispatch
//! - Struct-size-safe initialization
//! - Error handling and conversion
//!
//! This module does NOT manage PJRT objects--see types.zig for that.
const std = @import("std");
const c = @import("c.zig").c;

pub fn pjrt_struct_size(comptime T: type) usize {
    const maybe_struct_name: ?[]const u8 = comptime blk: {
        const needle = ".struct_";
        const type_name = @typeName(T);
        const idx = std.mem.indexOf(u8, type_name, needle) orelse break :blk null;
        break :blk type_name[idx + needle.len ..];
    };
    const struct_name = maybe_struct_name orelse return @sizeOf(T);
    const size_decl_name = comptime struct_name ++ "_STRUCT_SIZE";
    if (!@hasDecl(c, size_decl_name)) return @sizeOf(T);
    return @field(c, size_decl_name);
}

pub const Api = struct {
    handle: *anyopaque,
    pjrt_api: *c.PJRT_Api,
    trace_execute: bool,

    pub const InitOptions = struct {
        /// Trace PJRT execute calls.
        trace_execute: bool = false,
    };

    /// Load the API from a dynamic-library handle.
    pub fn init(
        handle: *anyopaque,
        get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api,
        options: InitOptions,
    ) !Api {
        const pjrt_api = get_api_fn() orelse return error.GetApiFailed;
        return Api{
            .handle = handle,
            .pjrt_api = @constCast(pjrt_api),
            .trace_execute = options.trace_execute,
        };
    }

    /// Get API version
    pub fn version(self: *const Api) struct { major: usize, minor: usize } {
        const ver = self.pjrt_api.pjrt_api_version;
        return .{
            .major = @intCast(ver.major_version),
            .minor = @intCast(ver.minor_version),
        };
    }

    pub fn find_extension(self: *const Api, extension_type: c.PJRT_Extension_Type) ?*c.PJRT_Extension_Base {
        var ext = self.pjrt_api.extension_start;
        while (ext) |current| {
            const current_ptr: *c.PJRT_Extension_Base = @ptrCast(current);
            if (current_ptr.type == extension_type) return current_ptr;
            ext = current_ptr.next;
        }
        return null;
    }

    /// Resolve the PJRT FFI extension from the extension chain.
    pub fn ffi_extension(self: *const Api) ?*c.PJRT_FFI {
        const ext = self.find_extension(c.PJRT_Extension_Type_FFI) orelse return null;
        const ffi_ext: *c.PJRT_FFI = @ptrCast(@alignCast(ext));
        if (ffi_ext.register_handler == null) return null;
        return ffi_ext;
    }

    pub fn gpu_custom_call_extension(self: *const Api) ?*c.PJRT_Gpu_Custom_Call {
        const ext = self.find_extension(c.PJRT_Extension_Type_Gpu_Custom_Call) orelse return null;
        const gpu_ext: *c.PJRT_Gpu_Custom_Call = @ptrCast(@alignCast(ext));
        if (gpu_ext.custom_call == null) return null;
        return gpu_ext;
    }

    /// Reflection-based call wrapper
    ///
    /// Usage:
    ///   try api.call("PJRT_Client_Create", .{ .client = &client_ptr });
    ///
    /// TODO(pjrt): Replace raw error codes with a comptime-generated enum.
    pub fn call(
        self: *Api,
        comptime func_name: []const u8,
        args: anytype,
    ) !void {
        const func_ptr = @field(self.pjrt_api, func_name) orelse return error.FunctionNotAvailable;
        const pjrt_err = func_ptr(args);

        if (pjrt_err) |err| {
            const pjrt_error = PjrtError.from_handle(self, err);
            const msg = pjrt_error.get_message(std.heap.page_allocator) catch "Unable to get error message";
            defer if (msg.ptr != "Unable to get error message".ptr) std.heap.page_allocator.free(msg);
            std.debug.print("PJRT Error in {s}: {s}\n", .{ func_name, msg });
            return try pjrt_error.to_zig_error();
        }
    }

    /// Get function argument type by name
    fn FunctionArgsType(comptime func_name: []const u8) type {
        const args_name = func_name ++ "_Args";
        return @field(c, args_name);
    }
};

/// Initialize PJRT args struct with struct_size and priv
///
/// This only sets the boilerplate fields. Populate other fields manually
/// to avoid pointer-kind coercion issues in the generic.
///
/// Usage:
///   var args = init_args(c.PJRT_Client_Create_Args);
///   args.client = &client_ptr;
///
pub fn init_args(comptime Args: type) Args {
    var a: Args = std.mem.zeroes(Args);

    if (@hasField(Args, "struct_size")) {
        a.struct_size = pjrt_struct_size(Args);
    }
    if (@hasField(Args, "extension_start")) {
        @field(a, "extension_start") = null;
    }
    if (@hasField(Args, "priv")) {
        a.priv = null;
    }
    return a;
}

/// Helper for PJRT output parameters
///
/// Many PJRT functions return pointers via `[*c]?*T` out-params.
/// This helper provides a 1-element array slot that coerces correctly.
///
/// Usage:
///   var out_exe = Out1(c.PJRT_LoadedExecutable){};
///   args.executable = out_exe.out();
///   try api.call(...);
///   const exe = out_exe.get() orelse return error.NullReturn;
///
pub fn Out1(comptime T: type) type {
    return struct {
        slot: [1]?*T = .{null},

        pub fn out(self: *@This()) [*c]?*T {
            return &self.slot;
        }

        pub fn get(self: *@This()) ?*T {
            return self.slot[0];
        }
    };
}

pub const Error = error{
    /// PJRT_Error_Code_CANCELLED = 1
    Cancelled,
    /// PJRT_Error_Code_UNKNOWN = 2
    UnknownPjrtError,
    /// PJRT_Error_Code_INVALID_ARGUMENT = 3
    InvalidArgument,

    /// PJRT_Error_Code_DEADLINE_EXCEEDED = 4
    DeadlineExceeded,

    /// PJRT_Error_Code_NOT_FOUND = 5
    NotFound,

    /// PJRT_Error_Code_ALREADY_EXISTS = 6
    AlreadyExists,

    /// PJRT_Error_Code_PERMISSION_DENIED = 7
    PermissionDenied,

    /// PJRT_Error_Code_RESOURCE_EXHAUSTED = 8
    ResourceExhausted,

    /// PJRT_Error_Code_FAILED_PRECONDITION = 9
    FailedPrecondition,

    /// PJRT_Error_Code_ABORTED = 10
    Aborted,

    /// PJRT_Error_Code_OUT_OF_RANGE = 11
    OutOfRange,

    /// PJRT_Error_Code_UNIMPLEMENTED = 12
    Unimplemented,

    /// PJRT_Error_Code_INTERNAL = 13
    Internal,

    /// PJRT_Error_Code_UNAVAILABLE = 14
    Unavailable,

    /// PJRT_Error_Code_DATA_LOSS = 15
    DataLoss,

    /// PJRT_Error_Code_UNAUTHENTICATED = 16
    Unauthenticated,
} || std.mem.Allocator.Error;

/// PJRT Error wrapper with Zig error conversion
pub const PjrtError = struct {
    api: *Api,
    pjrt_error: *c.PJRT_Error,

    pub fn from_handle(api: *Api, pjrt_error: *c.PJRT_Error) PjrtError {
        return .{ .api = api, .pjrt_error = pjrt_error };
    }

    pub fn get_message(self: PjrtError, allocator: std.mem.Allocator) ![]const u8 {
        var args = init_args(c.PJRT_Error_Message_Args);
        args.@"error" = self.pjrt_error;

        const msg_fn = self.api.pjrt_api.PJRT_Error_Message orelse return error.FunctionNotAvailable;
        _ = msg_fn(&args);

        const msg = args.message[0..args.message_size];
        return try allocator.dupe(u8, msg);
    }

    pub fn get_code(self: PjrtError) !i32 {
        var args = init_args(c.PJRT_Error_GetCode_Args);
        args.@"error" = self.pjrt_error;

        const code_fn = self.api.pjrt_api.PJRT_Error_GetCode orelse return error.FunctionNotAvailable;
        _ = code_fn(&args);

        return @intCast(args.code);
    }

    pub fn to_zig_error(self: PjrtError) Error!void {
        const code = self.get_code() catch return Error.UnknownPjrtError;
        self.deinit();

        return switch (code) {
            // PJRT_Error_Code_OK = 0
            0 => {},

            // PJRT_Error_Code_CANCELLED = 1
            1 => Error.Cancelled,

            // PJRT_Error_Code_UNKNOWN = 2
            2 => Error.UnknownPjrtError,

            // PJRT_Error_Code_INVALID_ARGUMENT = 3
            3 => Error.InvalidArgument,

            // PJRT_Error_Code_DEADLINE_EXCEEDED = 4
            4 => Error.DeadlineExceeded,

            // PJRT_Error_Code_NOT_FOUND = 5
            5 => Error.NotFound,

            // PJRT_Error_Code_ALREADY_EXISTS = 6
            6 => Error.AlreadyExists,

            // PJRT_Error_Code_PERMISSION_DENIED = 7
            7 => Error.PermissionDenied,

            // PJRT_Error_Code_RESOURCE_EXHAUSTED = 8
            8 => Error.ResourceExhausted,

            // PJRT_Error_Code_FAILED_PRECONDITION = 9
            9 => Error.FailedPrecondition,

            // PJRT_Error_Code_ABORTED = 10
            10 => Error.Aborted,

            // PJRT_Error_Code_OUT_OF_RANGE = 11
            11 => Error.OutOfRange,

            // PJRT_Error_Code_UNIMPLEMENTED = 12
            12 => Error.Unimplemented,

            // PJRT_Error_Code_INTERNAL = 13
            13 => Error.Internal,

            // PJRT_Error_Code_UNAVAILABLE = 14
            14 => Error.Unavailable,

            // PJRT_Error_Code_DATA_LOSS = 15
            15 => Error.DataLoss,

            // PJRT_Error_Code_UNAUTHENTICATED = 16
            16 => Error.Unauthenticated,

            else => Error.UnknownPjrtError,
        };
    }

    pub fn deinit(self: PjrtError) void {
        var args = init_args(c.PJRT_Error_Destroy_Args);
        args.@"error" = self.pjrt_error;
        if (self.api.pjrt_api.PJRT_Error_Destroy) |destroy_fn| {
            _ = destroy_fn(&args);
        }
    }
};

test "init_args sets struct_size" {
    const TestStruct = struct {
        struct_size: usize,
        priv: ?*anyopaque,
        value: i32,
    };

    var s = init_args(TestStruct);
    s.value = 42;
    try std.testing.expectEqual(@sizeOf(TestStruct), s.struct_size);
    try std.testing.expectEqual(@as(i32, 42), s.value);
}
