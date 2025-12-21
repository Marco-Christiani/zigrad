/// PJRT Plugin Loader
///
/// Loads PJRT plugins from explicit paths using dlopen.
/// No Bazel runfiles, no environment mutation, no platform detection.
///
/// Usage:
///   const api = try loadPlugin("/path/to/pjrt_cpu_plugin.so");
///   defer unloadPlugin(api);
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c_mod = @import("c.zig");
const c = c_mod.c;

/// Load PJRT plugin from explicit path
///
/// Steps:
/// 1. dlopen(path, RTLD_NOW | RTLD_LOCAL)
/// 2. dlsym(handle, "GetPjrtApi")
/// 3. Call GetPjrtApi() to get PJRT_Api*
/// 4. Optionally call PJRT_Plugin_Initialize
///
pub fn loadPlugin(path: []const u8) !Api {
    // Null-terminate path for C
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try std.fmt.bufPrintZ(&path_buf, "{s}", .{path});

    // Open plugin library
    const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_LOCAL) orelse {
        const err_msg = c.dlerror();
        std.debug.print("dlopen failed: {s}\n", .{std.mem.span(err_msg)});
        return error.PluginLoadFailed;
    };
    errdefer _ = c.dlclose(handle);

    // Lookup GetPjrtApi symbol
    const get_api_sym = c.dlsym(handle, "GetPjrtApi") orelse {
        const err_msg = c.dlerror();
        std.debug.print("dlsym(GetPjrtApi) failed: {s}\n", .{std.mem.span(err_msg)});
        return error.SymbolNotFound;
    };

    const get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api = @ptrCast(@alignCast(get_api_sym));

    // Initialize API wrapper
    var api = try Api.init(handle, get_api_fn);

    // Optional: Call PJRT_Plugin_Initialize if present
    if (c.dlsym(handle, "PJRT_Plugin_Initialize")) |init_sym| {
        const init_fn: *const fn (*c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error = @ptrCast(@alignCast(init_sym));

        var init_args = api_mod.initArgs(c.PJRT_Plugin_Initialize_Args);
        if (init_fn(&init_args)) |pjrt_err| {
            const pjrt_error = api_mod.PjrtError.fromHandle(&api, pjrt_err);
            defer pjrt_error.deinit();
            return error.PluginInitFailed;
        }
    }

    // Validate API version
    const ver = api.version();
    if (ver.major == 0 and ver.minor < 40) {
        std.debug.print("Warning: PJRT API version {}.{} is old (expected >= 0.40)\n", .{ ver.major, ver.minor });
    }

    return api;
}

/// Unload PJRT plugin
pub fn unloadPlugin(api: Api) void {
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
