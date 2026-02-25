const std = @import("std");
const c = @import("c.zig");

const log = std.log.scoped(.@"zg/mirage_api");

pub const MirageError = error{
    MirageUnavailable,
    MirageCompileFailed,
    MirageExecuteFailed,
    OutOfMemory,
};

const RTLD_NOW: c_int = 0x2;
const RTLD_GLOBAL: c_int = 0x100;
extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;

var runtime_lib_handle: ?*anyopaque = null;

pub fn ensure_loaded() MirageError!void {
    if (runtime_lib_handle == null) {
        runtime_lib_handle = dlopen("libmirage_runtime.so", RTLD_NOW | RTLD_GLOBAL);
        if (runtime_lib_handle == null) {
            if (dlerror()) |err| {
                log.err("dlopen(libmirage_runtime.so) failed: {s}", .{std.mem.span(err)});
            }
            return error.MirageUnavailable;
        }
    }

    c.ensure_loaded(runtime_lib_handle.?) catch {
        return error.MirageUnavailable;
    };
}

pub fn to_status_error(status: c.MirageStatus) MirageError!void {
    return switch (status) {
        .ok => {},
        .invalid_argument, .unsupported => error.MirageCompileFailed,
        .internal_error => error.MirageExecuteFailed,
    };
}

pub fn compile_status_error(status: c.MirageStatus) MirageError!void {
    return switch (status) {
        .ok => {},
        .unsupported => error.MirageCompileFailed,
        .invalid_argument, .internal_error => error.MirageCompileFailed,
    };
}

pub fn execute_status_error(status: c.MirageStatus) MirageError!void {
    return switch (status) {
        .ok => {},
        .unsupported, .invalid_argument, .internal_error => error.MirageExecuteFailed,
    };
}

pub fn status_name(status: c.MirageStatus) []const u8 {
    return std.mem.span(c.mirage_status_string(status));
}

pub const Context = struct {
    raw: ?*c.MirageContext,

    pub fn init() MirageError!Context {
        try ensure_loaded();
        var raw: ?*c.MirageContext = null;
        const status = c.mirage_context_create(&raw);
        if (status != .ok) {
            log.err("mirage_context_create failed: {s}", .{status_name(status)});
            return error.MirageUnavailable;
        }
        return .{ .raw = raw };
    }

    pub fn deinit(self: *Context) void {
        c.mirage_context_destroy(self.raw);
        self.raw = null;
    }
};
