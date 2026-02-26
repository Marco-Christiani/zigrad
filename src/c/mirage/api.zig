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
var runtime_load_mutex: std.Thread.Mutex = .{};

pub fn ensure_loaded() MirageError!void {
    runtime_load_mutex.lock();
    defer runtime_load_mutex.unlock();

    if (runtime_lib_handle == null) {
        runtime_lib_handle = try load_runtime_library();
    }

    c.ensure_loaded(runtime_lib_handle.?) catch {
        return error.MirageUnavailable;
    };
}

fn load_runtime_library() MirageError!*anyopaque {
    if (dlopen("libmirage_runtime.so", RTLD_NOW | RTLD_GLOBAL)) |h| {
        return h;
    } else if (dlerror()) |err| {
        log.debug("dlopen(libmirage_runtime.so) failed: {s}", .{std.mem.span(err)});
    }

    const allocator = std.heap.smp_allocator;

    var has_static_archive = false;

    const runtime_handle = try load_runtime_from_sdk_root(allocator, "ZG_RUNTIME_SDK_ROOT", &has_static_archive);
    if (runtime_handle) |h| return h;

    const external_handle = try load_runtime_from_sdk_root(allocator, "ZG_EXTERNAL_SDK_ROOT", &has_static_archive);
    if (external_handle) |h| return h;

    if (has_static_archive) {
        log.err("mirage runtime shared library missing; found static archive only (libmirage_runtime.a)", .{});
    } else {
        log.err("mirage runtime shared library not found (libmirage_runtime.so)", .{});
    }
    return error.MirageUnavailable;
}

fn load_runtime_from_sdk_root(
    allocator: std.mem.Allocator,
    comptime env_name: []const u8,
    has_static_archive: *bool,
) MirageError!?*anyopaque {
    const sdk_root = std.process.getEnvVarOwned(allocator, env_name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.MirageUnavailable,
    };
    defer allocator.free(sdk_root);

    const so_path = try std.fmt.allocPrint(allocator, "{s}/lib/libmirage_runtime.so", .{sdk_root});
    defer allocator.free(so_path);

    const so_path_z = try allocator.allocSentinel(u8, so_path.len, 0);
    defer allocator.free(so_path_z);
    @memcpy(so_path_z[0..so_path.len], so_path);

    if (dlopen(so_path_z.ptr, RTLD_NOW | RTLD_GLOBAL)) |h| {
        return h;
    } else if (dlerror()) |err| {
        log.debug("dlopen({s}) failed: {s}", .{ so_path, std.mem.span(err) });
    }

    const a_path = try std.fmt.allocPrint(allocator, "{s}/lib/libmirage_runtime.a", .{sdk_root});
    defer allocator.free(a_path);
    std.fs.accessAbsolute(a_path, .{}) catch return null;
    has_static_archive.* = true;
    return null;
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
