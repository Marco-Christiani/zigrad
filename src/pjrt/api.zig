/// PJRT C-API Bindings
///
/// Low-level wrapper around PJRT C-API with:
/// - Reflection-based call dispatch (inspired by ZML pattern)
/// - Struct-size-safe initialization
/// - Error handling and conversion
///
/// This module does NOT manage PJRT objects—see types.zig for that.
const std = @import("std");
const c_mod = @import("c.zig");
const c = c_mod.c;

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

    /// Reflection-based call wrapper (ZML pattern)
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
        a.struct_size = @sizeOf(Args);
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

        // Map PJRT error codes to Zig errors
        // For now, use generic errors—refine with actual PJRT error codes
        return switch (code) {
            0 => {}, // Success (shouldn't happen)
            1 => error.InvalidArgument,
            2 => error.NotFound,
            3 => error.AlreadyExists,
            4 => error.ResourceExhausted,
            5 => error.Unimplemented,
            6 => error.Internal,
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
