/// PJRT Plugin Loader
///
/// Loads PJRT plugins from explicit paths using dlopen.
///
/// Usage:
///   const api = try load_plugin("/path/to/pjrt_cpu_plugin.so");
///   defer unload_plugin(api);
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c_mod = @import("c.zig");
const c = c_mod.c;
const DlHandle = ?*anyopaque;

var cached_api: ?Api = null;
var cached_path: ?[]const u8 = null;
var cached_refcount: usize = 0;

fn dl_err_msg() []const u8 {
    // dlerror() can return null
    const p = c.dlerror() orelse return "dlerror() returned null";
    return std.mem.span(p);
}

fn debug_enabled() bool {
    const allocator = std.heap.page_allocator;
    if (std.process.getEnvVarOwned(allocator, "ZG_PJRT_DEBUG")) |val| {
        defer allocator.free(val);
        if (val.len == 0) return false;
        return val[0] != '0';
    } else |_| {
        return false;
    }
}

fn dlclose_enabled() bool {
    const allocator = std.heap.page_allocator;
    if (std.process.getEnvVarOwned(allocator, "ZG_PJRT_SKIP_DLCLOSE")) |val| {
        defer allocator.free(val);
        if (val.len == 0) return false;
        if (val[0] != '0') return false;
    } else |_| {}

    if (std.process.getEnvVarOwned(allocator, "ZG_PJRT_DLCLOSE")) |val| {
        defer allocator.free(val);
        if (val.len == 0) return false;
        return val[0] != '0';
    } else |_| {
        return false;
    }
}

fn log_dladdr(label: []const u8, addr: *const anyopaque) void {
    if (!debug_enabled()) return;
    var info: c.Dl_info = undefined;
    if (c.dladdr(addr, &info) == 0) {
        std.debug.print("[pjrt-debug] dladdr {s}: <unresolved> addr={*}\n", .{ label, addr });
        return;
    }
    const fname = if (info.dli_fname) |p| std.mem.span(p) else "<null>";
    const sname = if (info.dli_sname) |p| std.mem.span(p) else "<null>";
    std.debug.print("[pjrt-debug] dladdr {s}: dso={s} sym={s} addr={*}\n", .{ label, fname, sname, addr });
}

fn canonicalize_path(path: []const u8) ![]const u8 {
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
pub fn load_plugin(path: []const u8) !Api {
    const debug = debug_enabled();

    const canonical = try canonicalize_path(path);
    errdefer std.heap.page_allocator.free(canonical);

    // GPU plugins typically require host-injected NVIDIA driver libs (e.g. libcuda.so.1).
    // CPU plugins should not hard-require them.
    const base = std.fs.path.basename(canonical);
    const wants_cuda = std.mem.indexOf(u8, base, "gpu") != null or std.mem.indexOf(u8, base, "cuda") != null;
    if (wants_cuda) {
        _ = try preload_host_nvidia(true);
    } else {
        _ = preload_host_nvidia(false) catch {};
    }

    if (cached_path) |existing| {
        if (!std.mem.eql(u8, existing, canonical)) {
            if (debug) {
                std.debug.print("[pjrt-debug] load_plugin called with different path\n", .{});
                std.debug.print("[pjrt-debug] existing={s}\n", .{existing});
                std.debug.print("[pjrt-debug] requested={s}\n", .{canonical});
            }
            return error.PluginAlreadyLoaded;
        }
        cached_refcount += 1;
        if (debug) {
            std.debug.print("[pjrt-debug] load_plugin reuse: path={s} handle={*} refcount={}\n", .{
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
        std.debug.print("dlopen failed: {s}\n", .{dl_err_msg()});

        // Optional retry: some plugins assume global symbol visibility
        const handle2 = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
            std.debug.print("dlopen retry (GLOBAL) failed: {s}\n", .{dl_err_msg()});
            return error.PluginLoadFailed;
        };
        const api = try load_from_handle(handle2, canonical);
        cached_api = api;
        cached_path = canonical;
        cached_refcount = 1;
        return api;
    };

    const api = try load_from_handle(handle, canonical);
    cached_api = api;
    cached_path = canonical;
    cached_refcount = 1;
    return api;
}

// ----------------------------------------------------------------------------------------------------

fn maybe_dlopen(
    label: []const u8,
    soname: [:0]const u8,
    flags: c_int,
    verbose: bool,
) ?*anyopaque {
    _ = c.dlerror(); // clear
    const h = c.dlopen(soname.ptr, flags);
    if (h == null) {
        if (verbose) {
            std.debug.print("dlopen FAIL {s}: {s} -> {s}\n", .{ label, soname, dl_err_msg() });
        }
        return null;
    }
    if (verbose) {
        std.debug.print("dlopen OK   {s}: {s}\n", .{ label, soname });
    }
    return h;
}

pub const HostNvidiaHandles = struct {
    cuda: *anyopaque,
    nvml: ?*anyopaque,
};

pub const PreloadError = error{
    CudaDriverNotFound,
};

/// Preload host-injected NVIDIA driver libraries.
/// Contract:
/// - Requires: libcuda.so.1 (stable soname)
/// - Optional: libnvidia-ml.so.1 (NVML)
/// This assumes your build/install/runtime has arranged for the dynamic loader
/// to find these (e.g. Docker GPU injection, or on NixOS: /run/opengl-driver/lib
/// in RUNPATH).
pub fn preload_host_nvidia(verbose: bool) PreloadError!HostNvidiaHandles {
    // RTLD_GLOBAL is often important for driver-side symbol visibility when
    // downstream DSOs expect to resolve CUDA driver symbols.
    const flags_driver: c_int = c.RTLD_NOW | c.RTLD_GLOBAL;

    const cuda_h = maybe_dlopen("CUDA driver", "libcuda.so.1", flags_driver, verbose) orelse {
        // No absolute-path fallback here: if this fails, it’s an environment/packaging issue.
        if (!verbose) {
            std.debug.print("dlopen FAIL CUDA driver (libcuda.so.1) -> {s}\n", .{dl_err_msg()});
        }
        return error.CudaDriverNotFound;
    };

    const nvml_h = maybe_dlopen("NVML", "libnvidia-ml.so.1", flags_driver, verbose);

    return .{
        .cuda = cuda_h,
        .nvml = nvml_h,
    };
}

// ----------------------------------------------------------------------------------------------------

fn load_from_handle(handle: *anyopaque, canonical_path: []const u8) !Api {
    errdefer _ = c.dlclose(handle);

    const get_api_sym = c.dlsym(handle, "GetPjrtApi") orelse {
        std.debug.print("dlsym(GetPjrtApi) failed: {s}\n", .{dl_err_msg()});
        return error.SymbolNotFound;
    };

    const get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api =
        @ptrCast(@alignCast(get_api_sym));

    if (debug_enabled()) {
        std.debug.print("[pjrt-debug] dlopen path={s} handle={*}\n", .{ canonical_path, handle });
        log_dladdr("GetPjrtApi", @ptrCast(@constCast(get_api_sym)));
    }

    var api = try Api.init(handle, get_api_fn);

    // if (c.dlsym(handle, "PJRT_Plugin_Initialize")) |init_sym| {
    //     const init_fn: *const fn (*c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error =
    //         @ptrCast(@alignCast(init_sym));
    //
    //     var init_args = api_mod.init_args(c.PJRT_Plugin_Initialize_Args);
    //     if (init_fn(&init_args)) |pjrt_err| {
    //         const pjrt_error = api_mod.PjrtError.from_handle(&api, pjrt_err);
    //         defer pjrt_error.deinit();
    //         return error.PluginInitFailed;
    //     }
    // }

    if (@field(api.pjrt_api, "PJRT_Plugin_Initialize")) |_| {
        var init_args = api_mod.init_args(c.PJRT_Plugin_Initialize_Args);
        try api.call("PJRT_Plugin_Initialize", &init_args);
    }

    if (debug_enabled()) {
        std.debug.print("[pjrt-debug] PJRT_Api ptr={*} extension_start={*}\n", .{
            api.pjrt_api,
            api.pjrt_api.extension_start,
        });
        if (api.pjrt_api.extension_start) |ext_ptr| {
            log_dladdr("PJRT_Api extension_start", @ptrCast(@constCast(ext_ptr)));
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
///
/// By default we keep the plugin loaded for process lifetime (skip `dlclose`),
/// since some PJRT plugins can crash on unload after exercising certain paths
/// (observed with `PJRT_Executable_DeserializeAndLoad`).
///
/// Set `ZG_PJRT_DLCLOSE=1` to enable `dlclose` on final unload.
pub fn unload_plugin(api: Api) void {
    const do_dlclose = dlclose_enabled();
    if (cached_api) |cached| {
        if (cached.handle == api.handle) {
            if (cached_refcount > 0) cached_refcount -= 1;
            if (debug_enabled()) {
                std.debug.print("[pjrt-debug] unload_plugin handle={*} refcount={}\n", .{
                    api.handle,
                    cached_refcount,
                });
            }
            if (cached_refcount == 0 and do_dlclose) {
                _ = c.dlclose(api.handle);
                if (cached_path) |p| std.heap.page_allocator.free(p);
                cached_api = null;
                cached_path = null;
            }
            return;
        }
    }
    if (do_dlclose) _ = c.dlclose(api.handle);
}

/// Get plugin path from environment or use default
pub fn get_plugin_path(allocator: std.mem.Allocator, backend_name: []const u8) ![]const u8 {
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
