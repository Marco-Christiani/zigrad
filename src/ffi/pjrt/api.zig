/// PJRT C-API Bindings
///
/// Low-level wrapper around PJRT C-API with:
/// - Reflection-based call dispatch
/// - Struct-size-safe initialization
/// - Error handling and conversion
///
/// This module does NOT manage PJRT objects—see types.zig for that.
const std = @import("std");
const c_mod = @import("c.zig");
const c = c_mod.c;

pub fn pjrtStructSize(comptime T: type) usize {
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

    /// Load API from dlopen handle
    pub fn init(handle: *anyopaque, get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api) !Api {
        const pjrt_api = get_api_fn() orelse return error.GetApiFailed;
        return Api{
            .handle = handle,
            .pjrt_api = @constCast(pjrt_api),
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

    /// Reflection-based call wrapper
    ///
    /// Usage:
    ///   try api.call("PJRT_Client_Create", .{ .client = &client_ptr });
    ///
    pub fn call(
        self: *Api,
        comptime func_name: []const u8,
        args: anytype,
    ) !void {
        const func_ptr = @field(self.pjrt_api, func_name) orelse return error.FunctionNotAvailable;
        const pjrt_err = func_ptr(args);

        if (pjrt_err) |err| {
            const pjrt_error = PjrtError.fromHandle(self, err);
            const msg = pjrt_error.getMessage(std.heap.page_allocator) catch "Unable to get error message";
            defer if (msg.ptr != "Unable to get error message".ptr) std.heap.page_allocator.free(msg);
            std.debug.print("PJRT Error in {s}: {s}\n", .{ func_name, msg });
            return try pjrt_error.toZigError();
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
///   var args = initArgs(c.PJRT_Client_Create_Args);
///   args.client = &client_ptr;
///
pub fn initArgs(comptime Args: type) Args {
    var a: Args = std.mem.zeroes(Args);

    if (@hasField(Args, "struct_size")) {
        a.struct_size = pjrtStructSize(Args);
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

/// PJRT Error wrapper with Zig error conversion
pub const PjrtError = struct {
    api: *Api,
    pjrt_error: *c.PJRT_Error,

    pub fn fromHandle(api: *Api, pjrt_error: *c.PJRT_Error) PjrtError {
        return .{ .api = api, .pjrt_error = pjrt_error };
    }

    pub fn getMessage(self: PjrtError, allocator: std.mem.Allocator) ![]const u8 {
        var args = initArgs(c.PJRT_Error_Message_Args);
        args.@"error" = self.pjrt_error;

        const msg_fn = self.api.pjrt_api.PJRT_Error_Message orelse return error.FunctionNotAvailable;
        _ = msg_fn(&args);

        const msg = args.message[0..args.message_size];
        return allocator.dupe(u8, msg);
    }

    pub fn getCode(self: PjrtError) !i32 {
        var args = initArgs(c.PJRT_Error_GetCode_Args);
        args.@"error" = self.pjrt_error;

        const code_fn = self.api.pjrt_api.PJRT_Error_GetCode orelse return error.FunctionNotAvailable;
        _ = code_fn(&args);

        return @intCast(args.code);
    }

    pub fn toZigError(self: PjrtError) !void {
        const code = self.getCode() catch return error.UnknownPjrtError;
        self.deinit();

        return switch (code) {
            // PJRT_Error_Code_OK = 0
            0 => {},

            // PJRT_Error_Code_CANCELLED = 1
            1 => error.Cancelled,

            // PJRT_Error_Code_UNKNOWN = 2
            2 => error.UnknownPjrtError,

            // PJRT_Error_Code_INVALID_ARGUMENT = 3
            3 => error.InvalidArgument,

            // PJRT_Error_Code_DEADLINE_EXCEEDED = 4
            4 => error.DeadlineExceeded,

            // PJRT_Error_Code_NOT_FOUND = 5
            5 => error.NotFound,

            // PJRT_Error_Code_ALREADY_EXISTS = 6
            6 => error.AlreadyExists,

            // PJRT_Error_Code_PERMISSION_DENIED = 7
            7 => error.PermissionDenied,

            // PJRT_Error_Code_RESOURCE_EXHAUSTED = 8
            8 => error.ResourceExhausted,

            // PJRT_Error_Code_FAILED_PRECONDITION = 9
            9 => error.FailedPrecondition,

            // PJRT_Error_Code_ABORTED = 10
            10 => error.Aborted,

            // PJRT_Error_Code_OUT_OF_RANGE = 11
            11 => error.OutOfRange,

            // PJRT_Error_Code_UNIMPLEMENTED = 12
            12 => error.Unimplemented,

            // PJRT_Error_Code_INTERNAL = 13
            13 => error.Internal,

            // PJRT_Error_Code_UNAVAILABLE = 14
            14 => error.Unavailable,

            // PJRT_Error_Code_DATA_LOSS = 15
            15 => error.DataLoss,

            // PJRT_Error_Code_UNAUTHENTICATED = 16
            16 => error.Unauthenticated,

            else => error.UnknownPjrtError,
        };
    }

    pub fn deinit(self: PjrtError) void {
        var args = initArgs(c.PJRT_Error_Destroy_Args);
        args.@"error" = self.pjrt_error;
        if (self.api.pjrt_api.PJRT_Error_Destroy) |destroy_fn| {
            _ = destroy_fn(&args);
        }
    }
};

test "initArgs sets struct_size" {
    const TestStruct = struct {
        struct_size: usize,
        priv: ?*anyopaque,
        value: i32,
    };

    var s = initArgs(TestStruct);
    s.value = 42;
    try std.testing.expectEqual(@sizeOf(TestStruct), s.struct_size);
    try std.testing.expectEqual(@as(i32, 42), s.value);
}
