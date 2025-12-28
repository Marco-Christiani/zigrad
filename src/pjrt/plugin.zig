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
const DlHandle = ?*anyopaque;

fn dlErrMsg() []const u8 {
    // dlerror() can return null
    const p = c.dlerror() orelse return "dlerror() returned null";
    return std.mem.span(p);
}

/// Preload one of the candidate libraries into the global namespace.
/// This avoids relying on LD_LIBRARY_PATH and avoids Nix/glibc search-path weirdness.
fn preloadGlobal(candidates: []const []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;

    for (candidates) |cand| {
        const cand_z = std.fmt.bufPrintZ(&buf, "{s}", .{cand}) catch continue;

        _ = c.dlerror(); // clear
        const h: DlHandle = c.dlopen(cand_z, c.RTLD_NOW | c.RTLD_GLOBAL);
        if (h != null) {
            std.debug.print("preloaded: {s}\n", .{cand});
            return;
        }
    }

    // If all failed, print last dlerror for debugging (but do not hard-fail here).
    std.debug.print("preload failed (last): {s}\n", .{dlErrMsg()});
}

fn probeCudaDriver() void {
    const CUresult = c_uint;
    // Resolve symbols from the global namespace (libcuda preloaded RTLD_GLOBAL).
    const cuInit_sym = c.dlsym(null, "cuInit") orelse {
        std.debug.print("probe: dlsym(cuInit) failed: {s}\n", .{dlErrMsg()});
        return;
    };
    const cuDeviceGetCount_sym = c.dlsym(null, "cuDeviceGetCount") orelse {
        std.debug.print("probe: dlsym(cuDeviceGetCount) failed: {s}\n", .{dlErrMsg()});
        return;
    };

    const cuInit: *const fn (c_uint) callconv(.c) CUresult = @ptrCast(@alignCast(cuInit_sym));
    const cuDeviceGetCount: *const fn (*c_int) callconv(.c) CUresult = @ptrCast(@alignCast(cuDeviceGetCount_sym));

    const r0 = cuInit(0);
    std.debug.print("probe: cuInit -> {d}\n", .{r0});

    var n: c_int = -1;
    const r1 = cuDeviceGetCount(&n);
    std.debug.print("probe: cuDeviceGetCount -> {d}, n={d}\n", .{ r1, n });
}

fn preloadRuntimeDeps() void {
    // C++ runtime (needed by jaxlib / XLA pieces)
    preloadGlobal(&[_][]const u8{
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "libstdc++.so.6",
    });

    // Often required by libstdc++
    preloadGlobal(&[_][]const u8{
        "/lib/x86_64-linux-gnu/libgcc_s.so.1",
        "/usr/lib/x86_64-linux-gnu/libgcc_s.so.1",
        "libgcc_s.so.1",
    });

    // NVIDIA driver (must come from host driver inside container)
    preloadGlobal(&[_][]const u8{
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "libcuda.so.1",
    });

    preloadGlobal(&[_][]const u8{
        "/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "libnvidia-ml.so.1",
    });

    probeCudaDriver();
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
    preloadRuntimeDeps();

    // Null-terminate path for C
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try std.fmt.bufPrintZ(&path_buf, "{s}", .{path});

    // Open plugin library
    // const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_LOCAL) orelse {
    const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
        std.debug.print("dlopen failed: {s}\n", .{dlErrMsg()});

        // Optional retry: some plugins assume global symbol visibility
        const handle2 = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
            std.debug.print("dlopen retry (GLOBAL) failed: {s}\n", .{dlErrMsg()});
            return error.PluginLoadFailed;
        };
        return try loadFromHandle(handle2);
    };

    return try loadFromHandle(handle);
}

fn loadFromHandle(handle: *anyopaque) !Api {
    errdefer _ = c.dlclose(handle);

    const get_api_sym = c.dlsym(handle, "GetPjrtApi") orelse {
        std.debug.print("dlsym(GetPjrtApi) failed: {s}\n", .{dlErrMsg()});
        return error.SymbolNotFound;
    };

    const get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api =
        @ptrCast(@alignCast(get_api_sym));

    var api = try Api.init(handle, get_api_fn);

    if (c.dlsym(handle, "PJRT_Plugin_Initialize")) |init_sym| {
        const init_fn: *const fn (*c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error =
            @ptrCast(@alignCast(init_sym));

        var init_args = api_mod.initArgs(c.PJRT_Plugin_Initialize_Args);
        if (init_fn(&init_args)) |pjrt_err| {
            const pjrt_error = api_mod.PjrtError.fromHandle(&api, pjrt_err);
            defer pjrt_error.deinit();
            return error.PluginInitFailed;
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
