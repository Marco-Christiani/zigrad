/// PJRT Plugin Loader
///
/// Loads PJRT plugins from explicit paths using dlopen.
///
/// Usage:
///   const api = try loadPlugin("/path/to/pjrt_cpu_plugin.so");
///   defer unloadPlugin(api);
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c_mod = @import("c.zig");
const c = c_mod.c;
const DlHandle = ?*anyopaque;

var cached_api: ?Api = null;
var cached_path: ?[]const u8 = null;
var cached_refcount: usize = 0;

fn dlErrMsg() []const u8 {
    // dlerror() can return null
    const p = c.dlerror() orelse return "dlerror() returned null";
    return std.mem.span(p);
}

fn debugEnabled() bool {
    const allocator = std.heap.page_allocator;
    if (std.process.getEnvVarOwned(allocator, "ZG_PJRT_DEBUG")) |val| {
        defer allocator.free(val);
        if (val.len == 0) return false;
        return val[0] != '0';
    } else |_| {
        return false;
    }
}

fn logDladdr(label: []const u8, addr: *const anyopaque) void {
    if (!debugEnabled()) return;
    var info: c.Dl_info = undefined;
    if (c.dladdr(addr, &info) == 0) {
        std.debug.print("[pjrt-debug] dladdr {s}: <unresolved> addr={*}\n", .{ label, addr });
        return;
    }
    const fname = if (info.dli_fname) |p| std.mem.span(p) else "<null>";
    const sname = if (info.dli_sname) |p| std.mem.span(p) else "<null>";
    std.debug.print("[pjrt-debug] dladdr {s}: dso={s} sym={s} addr={*}\n", .{ label, fname, sname, addr });
}

fn canonicalizePath(path: []const u8) ![]const u8 {
    return std.fs.cwd().realpathAlloc(std.heap.page_allocator, path);
}

/// Load PJRT plugin from explicit path
///
/// Steps:
/// 1. dlopen(path, RTLD_NOW | RTLD_LOCAL)
/// 2. dlsym(handle, "GetPjrtApi")
/// 3. Call GetPjrtApi() to get PJRT_Api*
/// 4. Optionally call PJRT_Plugin_Initialize
///
pub fn loadPlugin(path: []const u8) !Api {
    // preloadDriver();
    probeCudnn("result/runtime");
    const debug = debugEnabled();

    const canonical = try canonicalizePath(path);
    errdefer std.heap.page_allocator.free(canonical);

    if (cached_path) |existing| {
        if (!std.mem.eql(u8, existing, canonical)) {
            if (debug) {
                std.debug.print("[pjrt-debug] loadPlugin called with different path\n", .{});
                std.debug.print("[pjrt-debug] existing={s}\n", .{existing});
                std.debug.print("[pjrt-debug] requested={s}\n", .{canonical});
            }
            return error.PluginAlreadyLoaded;
        }
        cached_refcount += 1;
        if (debug) {
            std.debug.print("[pjrt-debug] loadPlugin reuse: path={s} handle={*} refcount={}\n", .{
                existing,
                cached_api.?.handle,
                cached_refcount,
            });
        }
        std.heap.page_allocator.free(canonical);
        return cached_api.?;
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try std.fmt.bufPrintZ(&path_buf, "{s}", .{canonical});

    // Open plugin library
    // const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_LOCAL) orelse {
    const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
        std.debug.print("dlopen failed: {s}\n", .{dlErrMsg()});

        // Optional retry: some plugins assume global symbol visibility
        const handle2 = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
            std.debug.print("dlopen retry (GLOBAL) failed: {s}\n", .{dlErrMsg()});
            return error.PluginLoadFailed;
        };
        const api = try loadFromHandle(handle2, canonical);
        cached_api = api;
        cached_path = canonical;
        cached_refcount = 1;
        return api;
    };

    const api = try loadFromHandle(handle, canonical);
    cached_api = api;
    cached_path = canonical;
    cached_refcount = 1;
    return api;
}

fn preloadDriver() void {
    const flags = c.RTLD_NOW | c.RTLD_GLOBAL;

    const libs = [_][]const u8{
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        // "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    };

    for (libs) |p| {
        const handle = c.dlopen(p.ptr, flags);
        if (handle == null) {
            std.debug.print("dlopen failed for {s}\n", .{p});
        }
    }
}

// ----------------------------------------------------------------------------------------------------

fn probeDlopen(label: []const u8, path: [:0]const u8, flags: c_int) ?*anyopaque {
    _ = c.dlerror(); // clear
    const h = c.dlopen(path.ptr, flags);
    if (h == null) {
        std.debug.print("probe dlopen FAIL {s}: {s} -> {s}\n", .{ label, path, dlErrMsg() });
        return null;
    }
    std.debug.print("probe dlopen OK   {s}: {s}\n", .{ label, path });
    return h;
}

fn probeCudnn(runtime_root: []const u8) void {
    const flags = c.RTLD_NOW | c.RTLD_LOCAL;

    // host driver libs
    _ = probeDlopen("libcuda", "/usr/lib/x86_64-linux-gnu/libcuda.so.1", c.RTLD_NOW | c.RTLD_GLOBAL);
    _ = probeDlopen("libcuda", "/lib/x86_64-linux-gnu/libcuda.so.1", c.RTLD_NOW | c.RTLD_GLOBAL);
    _ = probeDlopen("nvml", "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", c.RTLD_NOW | c.RTLD_GLOBAL);
    _ = probeDlopen("nvml", "/lib/x86_64-linux-gnu/libnvidia-ml.so.1", c.RTLD_NOW | c.RTLD_GLOBAL);

    // rely on search path
    const h = blk: {
        const hh = probeDlopen("cudnn(name)", "libcudnn.so.9", flags);
        if (hh != null) break :blk hh;
        // try dlopen by absolute path into bundle (should always work if deps are present)
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const abs = std.fmt.bufPrintZ(&buf, "{s}/nvidia/cudnn/lib/libcudnn.so.9", .{runtime_root}) catch return;
        break :blk probeDlopen("cudnn(abs)", abs, flags) orelse return;
    };

    // 4) symbol check
    _ = c.dlerror();
    const sym = c.dlsym(h, "cudnnGetProperty");
    if (sym == null) {
        std.debug.print("probe dlsym FAIL cudnnGetProperty -> {s}\n", .{dlErrMsg()});
    } else {
        std.debug.print("probe dlsym OK   cudnnGetProperty\n", .{});
    }
}
// ----------------------------------------------------------------------------------------------------

fn loadFromHandle(handle: *anyopaque, canonical_path: []const u8) !Api {
    errdefer _ = c.dlclose(handle);

    const get_api_sym = c.dlsym(handle, "GetPjrtApi") orelse {
        std.debug.print("dlsym(GetPjrtApi) failed: {s}\n", .{dlErrMsg()});
        return error.SymbolNotFound;
    };

    const get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api =
        @ptrCast(@alignCast(get_api_sym));

    if (debugEnabled()) {
        std.debug.print("[pjrt-debug] dlopen path={s} handle={*}\n", .{ canonical_path, handle });
        logDladdr("GetPjrtApi", @ptrCast(@constCast(get_api_sym)));
    }

    var api = try Api.init(handle, get_api_fn);

    // if (c.dlsym(handle, "PJRT_Plugin_Initialize")) |init_sym| {
    //     const init_fn: *const fn (*c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error =
    //         @ptrCast(@alignCast(init_sym));
    //
    //     var init_args = api_mod.initArgs(c.PJRT_Plugin_Initialize_Args);
    //     if (init_fn(&init_args)) |pjrt_err| {
    //         const pjrt_error = api_mod.PjrtError.fromHandle(&api, pjrt_err);
    //         defer pjrt_error.deinit();
    //         return error.PluginInitFailed;
    //     }
    // }

    if (@field(api.pjrt_api, "PJRT_Plugin_Initialize")) |_| {
        var init_args = api_mod.initArgs(c.PJRT_Plugin_Initialize_Args);
        try api.call("PJRT_Plugin_Initialize", &init_args);
    }

    if (debugEnabled()) {
        std.debug.print("[pjrt-debug] PJRT_Api ptr={*} extension_start={*}\n", .{
            api.pjrt_api,
            api.pjrt_api.extension_start,
        });
        if (api.pjrt_api.extension_start) |ext_ptr| {
            logDladdr("PJRT_Api extension_start", @ptrCast(@constCast(ext_ptr)));
        }
    }

    const ver = api.version();
    std.debug.print("PJRT api version from plugin: {}.{}\n", .{ ver.major, ver.minor });
    std.debug.print("sizeof(PJRT_Client_Create_Args) = {}\n", .{@sizeOf(c.PJRT_Client_Create_Args)});

    if (ver.major == 0 and ver.minor < 40) {
        std.debug.print(
            "Warning: PJRT API version {}.{} is old (expected >= 0.40)\n",
            .{ ver.major, ver.minor },
        );
    }

    return api;
}

/// Unload PJRT plugin
pub fn unloadPlugin(api: Api) void {
    if (cached_api) |cached| {
        if (cached.handle == api.handle) {
            if (cached_refcount > 0) cached_refcount -= 1;
            if (debugEnabled()) {
                std.debug.print("[pjrt-debug] unloadPlugin handle={*} refcount={}\n", .{
                    api.handle,
                    cached_refcount,
                });
            }
            if (cached_refcount == 0) {
                _ = c.dlclose(api.handle);
                if (cached_path) |p| std.heap.page_allocator.free(p);
                cached_api = null;
                cached_path = null;
            }
            return;
        }
    }
    _ = c.dlclose(api.handle);
}

/// Get plugin path from environment or use default
pub fn getPluginPath(allocator: std.mem.Allocator, backend_name: []const u8) ![]const u8 {
    // Try environment variable first
    const env_var = try std.fmt.allocPrint(allocator, "PJRT_{s}_PLUGIN_PATH", .{backend_name});
    defer allocator.free(env_var);

    // Convert to uppercase
    for (env_var) |*ch| {
        ch.* = std.ascii.toUpper(ch.*);
    }

    if (std.process.getEnvVarOwned(allocator, env_var)) |path| {
        return path;
    } else |_| {
        // Fall back to default paths
        const default_name = try std.fmt.allocPrint(
            allocator,
            "libpjrt_{s}.so",
            .{backend_name},
        );
        defer allocator.free(default_name);

        // Try standard locations
        const search_paths = [_][]const u8{
            "/usr/local/lib",
            "/usr/lib",
            "./lib",
        };

        for (search_paths) |dir| {
            const full_path = try std.fs.path.join(allocator, &[_][]const u8{ dir, default_name });
            defer allocator.free(full_path);

            std.fs.accessAbsolute(full_path, .{}) catch continue;
            return try allocator.dupe(u8, full_path);
        }

        return error.PluginNotFound;
    }
}
