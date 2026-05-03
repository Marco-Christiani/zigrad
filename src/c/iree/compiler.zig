//! IREE Compiler bindings.
//!
//! Provides two compilation modes:
//!   1. **Subprocess** (default): invokes `iree-compile` as a child process.
//!      Avoids LLVM version conflicts when the host process also links LLVM
//!      (e.g. via the MLIR extension shim).
//!   2. **Dlopen**: loads `libIREECompiler.so` in-process via the embedding API.
//!      Only safe when no other LLVM-linked DSOs are loaded into the process.
//!
//! ## Usage (subprocess)
//!
//! ```zig
//! var cmp = Compiler.init_subprocess("/path/to/iree-compile", &.{"--iree-hal-target-backends=vmvx"});
//! const vmfb = try cmp.compile(allocator, mlir_bytes, is_bytecode);
//! defer allocator.free(vmfb);
//! ```
const std = @import("std");
const log = std.log.scoped(.@"zg/iree_compiler");

// ---------------------------------------------------------------------------
// Opaque handle types (match IREE's forward declarations).
// ---------------------------------------------------------------------------

pub const Session = opaque {};
pub const Invocation = opaque {};
pub const Source = opaque {};
pub const Output = opaque {};
pub const Error = opaque {};

/// Diagnostic severity levels (matches IreeCompilerDiagnosticSeverity enum).
pub const DiagnosticSeverity = enum(c_int) {
    note = 0,
    warning = 1,
    err = 2,
    remark = 3,
};

/// Pipeline type (matches IreeCompilerPipelineType enum).
pub const PipelineType = enum(c_int) {
    std = 0,
    hal_executable = 1,
    precompile = 2,
};

// ---------------------------------------------------------------------------
// Diagnostic callback type.
// ---------------------------------------------------------------------------

pub const DiagnosticCallback = *const fn (DiagnosticSeverity, [*c]const u8, usize, ?*anyopaque) callconv(.c) void;

// ---------------------------------------------------------------------------
// Function pointer table (resolved via dlsym after dlopen).
// ---------------------------------------------------------------------------

const Vtable = struct {
    get_api_version: *const fn () callconv(.c) c_int,
    global_initialize: *const fn () callconv(.c) void,
    global_shutdown: *const fn () callconv(.c) void,
    setup_global_cl: *const fn (c_int, [*c]const [*c]const u8, [*c]const u8, bool) callconv(.c) void,

    session_create: *const fn () callconv(.c) ?*Session,
    session_destroy: *const fn (?*Session) callconv(.c) void,
    session_set_flags: *const fn (?*Session, c_int, [*c]const [*c]const u8) callconv(.c) ?*Error,

    invocation_create: *const fn (?*Session) callconv(.c) ?*Invocation,
    invocation_enable_callback_diagnostics: *const fn (?*Invocation, c_int, DiagnosticCallback, ?*anyopaque) callconv(.c) void,
    invocation_destroy: *const fn (?*Invocation) callconv(.c) void,
    invocation_parse_source: *const fn (?*Invocation, ?*Source) callconv(.c) bool,
    invocation_pipeline: *const fn (?*Invocation, c_int) callconv(.c) bool,
    invocation_output_vm_bytecode: *const fn (?*Invocation, ?*Output) callconv(.c) ?*Error,

    source_wrap_buffer: *const fn (?*Session, [*c]const u8, [*c]const u8, usize, bool, *?*Source) callconv(.c) ?*Error,
    source_destroy: *const fn (?*Source) callconv(.c) void,

    output_open_membuffer: *const fn (*?*Output) callconv(.c) ?*Error,
    output_destroy: *const fn (?*Output) callconv(.c) void,
    output_map_memory: *const fn (?*Output, *?*anyopaque, *u64) callconv(.c) ?*Error,
    output_keep: *const fn (?*Output) callconv(.c) void,

    error_destroy: *const fn (?*Error) callconv(.c) void,
    error_get_message: *const fn (?*Error) callconv(.c) [*c]const u8,
};

fn resolve(comptime FnType: type, lib: *std.DynLib, name: [:0]const u8) !FnType {
    const sym = lib.lookup(*anyopaque, name) orelse {
        log.err("dlsym({s}) failed", .{name});
        return error.SymbolNotFound;
    };
    return @ptrCast(@alignCast(sym));
}

// ---------------------------------------------------------------------------
// Public Compiler handle.
// ---------------------------------------------------------------------------

/// IREE compiler handle supporting subprocess or in-process (dlopen) modes.
pub const Compiler = struct {
    mode: union(enum) {
        /// Subprocess mode: invoke iree-compile binary.
        subprocess: SubprocessConfig,
        /// Dlopen mode: in-process via embedding API.
        dlopen: DlopenState,
    },

    const SubprocessConfig = struct {
        /// Path to the `iree-compile` binary.
        exe_path: []const u8,
        /// Extra flags (e.g. `--iree-hal-target-backends=vmvx`).
        flags: []const []const u8,
    };

    const DlopenState = struct {
        lib: std.DynLib,
        vt: Vtable,
    };

    /// Create a subprocess-mode compiler.
    ///
    /// `exe_path` must point to the `iree-compile` binary.
    /// `flags` are appended to every invocation (e.g. target backend selection).
    pub fn init_subprocess(exe_path: []const u8, flags: []const []const u8) Compiler {
        log.info("IREE compiler (subprocess): {s}", .{exe_path});
        return .{ .mode = .{ .subprocess = .{ .exe_path = exe_path, .flags = flags } } };
    }

    /// Load `libIREECompiler.so` for in-process compilation.
    ///
    /// WARNING: crashes if the host process also loads LLVM shared libs
    /// (e.g. via the MLIR extension shim) due to LLVM version conflicts.
    pub fn init_dlopen(path: []const u8, cl_flags: ?[]const []const u8) !Compiler {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = std.fmt.bufPrintZ(&path_buf, "{s}", .{path}) catch return error.PathTooLong;

        var lib = std.DynLib.open(path_z) catch |err| {
            log.err("failed to open {s}: {s}", .{ path, @errorName(err) });
            return err;
        };

        const vt = load_vtable(&lib) catch |err| {
            lib.close();
            return err;
        };

        const ver = vt.get_api_version();
        log.info("IREE compiler API version: {d}.{d}", .{ ver >> 16, ver & 0xFFFF });

        vt.global_initialize();

        // SetupGlobalCL registers CL options (HAL targets, etc.).
        var argv_buf: [65][*c]const u8 = undefined;
        argv_buf[0] = "zigrad";
        var argc: c_int = 1;
        if (cl_flags) |flags| {
            for (flags) |f| {
                if (argc >= 65) break;
                argv_buf[@intCast(argc)] = f.ptr;
                argc += 1;
            }
        }
        vt.setup_global_cl(argc, &argv_buf, "zigrad", false);

        return .{ .mode = .{ .dlopen = .{ .lib = lib, .vt = vt } } };
    }

    /// Maps Vtable field names to their IREE C API symbol names.
    const symbol_names = .{
        .get_api_version = "ireeCompilerGetAPIVersion",
        .global_initialize = "ireeCompilerGlobalInitialize",
        .global_shutdown = "ireeCompilerGlobalShutdown",
        .setup_global_cl = "ireeCompilerSetupGlobalCL",
        .session_create = "ireeCompilerSessionCreate",
        .session_destroy = "ireeCompilerSessionDestroy",
        .session_set_flags = "ireeCompilerSessionSetFlags",
        .invocation_create = "ireeCompilerInvocationCreate",
        .invocation_enable_callback_diagnostics = "ireeCompilerInvocationEnableCallbackDiagnostics",
        .invocation_destroy = "ireeCompilerInvocationDestroy",
        .invocation_parse_source = "ireeCompilerInvocationParseSource",
        .invocation_pipeline = "ireeCompilerInvocationPipeline",
        .invocation_output_vm_bytecode = "ireeCompilerInvocationOutputVMBytecode",
        .source_wrap_buffer = "ireeCompilerSourceWrapBuffer",
        .source_destroy = "ireeCompilerSourceDestroy",
        .output_open_membuffer = "ireeCompilerOutputOpenMembuffer",
        .output_destroy = "ireeCompilerOutputDestroy",
        .output_map_memory = "ireeCompilerOutputMapMemory",
        .output_keep = "ireeCompilerOutputKeep",
        .error_destroy = "ireeCompilerErrorDestroy",
        .error_get_message = "ireeCompilerErrorGetMessage",
    };

    fn load_vtable(lib: *std.DynLib) !Vtable {
        var vt: Vtable = undefined;
        inline for (std.meta.fields(Vtable)) |field| {
            @field(vt, field.name) = try resolve(field.type, lib, @field(symbol_names, field.name));
        }
        return vt;
    }

    pub fn deinit(self: *Compiler) void {
        switch (self.mode) {
            .subprocess => {},
            .dlopen => |*s| {
                s.vt.global_shutdown();
                s.lib.close();
            },
        }
    }

    /// Compile `mlir_bytes` to VMFB (VM FlatBuffer) bytes.
    ///
    /// Caller owns the returned slice; free with `allocator.free`.
    pub fn compile(
        self: *const Compiler,
        io: std.Io,
        allocator: std.mem.Allocator,
        mlir_bytes: []const u8,
        is_bytecode: bool,
    ) ![]u8 {
        return switch (self.mode) {
            .subprocess => |cfg| compile_subprocess(io, allocator, cfg, mlir_bytes, is_bytecode),
            .dlopen => |s| compile_dlopen(&s, allocator, mlir_bytes, is_bytecode),
        };
    }
};

// ---------------------------------------------------------------------------
// Subprocess compilation.
// ---------------------------------------------------------------------------

fn compile_subprocess(
    io: std.Io,
    allocator: std.mem.Allocator,
    cfg: Compiler.SubprocessConfig,
    mlir_bytes: []const u8,
    is_bytecode: bool,
) ![]u8 {
    // Write MLIR to a temp file (iree-compile reads from file, not stdin for bytecode).
    const suffix: []const u8 = if (is_bytecode) ".mlirbc" else ".mlir";

    var tmp_input_path: [std.fs.max_path_bytes]u8 = undefined;
    var tmp_output_path: [std.fs.max_path_bytes]u8 = undefined;

    const in_path = try write_temp_file(io, mlir_bytes, suffix, &tmp_input_path);
    defer std.Io.Dir.cwd().deleteFile(io, in_path) catch {};

    // Build output path by replacing suffix.
    const out_path = try make_output_path(in_path, &tmp_output_path);
    defer std.Io.Dir.cwd().deleteFile(io, out_path) catch {};

    // Build argv: iree-compile <flags...> <input> -o <output>
    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    try argv.append(allocator, cfg.exe_path);
    for (cfg.flags) |f| try argv.append(allocator, f);
    try argv.append(allocator, in_path);
    try argv.append(allocator, "-o");
    try argv.append(allocator, out_path);

    log.debug("iree-compile: {d} bytes MLIR -> {s}", .{ mlir_bytes.len, out_path });

    const result = try std.process.run(allocator, io, .{
        .argv = argv.items,
        .expand_arg0 = .expand,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| {
            if (code != 0) {
                log.err("iree-compile exited with code {d}: {s}", .{ code, result.stderr });
                return error.CompileFailed;
            }
        },
        else => {
            log.err("iree-compile terminated abnormally", .{});
            return error.CompileFailed;
        },
    }

    // Read the VMFB output file.
    const vmfb = try std.Io.Dir.cwd().readFileAlloc(io, out_path, allocator, .limited(256 * 1024 * 1024));
    log.debug("compile: {d} bytes MLIR -> {d} bytes VMFB", .{ mlir_bytes.len, vmfb.len });
    return vmfb;
}

fn write_temp_file(
    io: std.Io,
    data: []const u8,
    suffix: []const u8,
    path_buf: *[std.fs.max_path_bytes]u8,
) ![]const u8 {
    // Try up to a few times with exclusive creation to avoid collisions.
    for (0..8) |_| {
        const ts = std.Io.Timestamp.now(io, .awake);
        const ns: i128 = ts.nanoseconds;
        const path = std.fmt.bufPrint(path_buf, "/tmp/zigrad-iree-{x}{s}", .{
            @as(u64, @truncate(@as(u128, @bitCast(ns)))),
            suffix,
        }) catch return error.PathTooLong;

        var file = std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true }) catch |err| {
            if (err == error.PathAlreadyExists) continue;
            return err;
        };
        defer file.close(io);
        try file.writeStreamingAll(io, data);
        return path;
    }
    return error.TempFileCollision;
}

fn make_output_path(
    in_path: []const u8,
    path_buf: *[std.fs.max_path_bytes]u8,
) ![]const u8 {
    // Replace extension with .vmfb
    const stem = if (std.mem.lastIndexOfScalar(u8, in_path, '.')) |dot|
        in_path[0..dot]
    else
        in_path;
    return std.fmt.bufPrint(path_buf, "{s}.vmfb", .{stem}) catch return error.PathTooLong;
}

// ---------------------------------------------------------------------------
// Dlopen compilation (in-process, embedding API).
// ---------------------------------------------------------------------------

fn compile_dlopen(
    state: *const Compiler.DlopenState,
    allocator: std.mem.Allocator,
    mlir_bytes: []const u8,
    is_bytecode: bool,
) ![]u8 {
    const vt = &state.vt;

    const session = vt.session_create() orelse return error.SessionCreateFailed;
    defer vt.session_destroy(session);

    const inv = vt.invocation_create(session) orelse return error.InvocationCreateFailed;
    defer vt.invocation_destroy(inv);

    // Install diagnostic handler (invocation-level in current API).
    var diag_ctx: DiagCtx = .{ .vt = vt };
    vt.invocation_enable_callback_diagnostics(inv, 0, diag_handler, @ptrCast(&diag_ctx));

    // Wrap the input MLIR buffer.
    var source: ?*Source = null;
    if (vt.source_wrap_buffer(
        session,
        "input",
        mlir_bytes.ptr,
        mlir_bytes.len,
        !is_bytecode,
        &source,
    )) |err| {
        defer vt.error_destroy(err);
        const msg = vt.error_get_message(err);
        log.err("IREE source_wrap_buffer failed: {s}", .{msg});
        return error.SourceWrapFailed;
    }
    defer vt.source_destroy(source);

    if (!vt.invocation_parse_source(inv, source)) {
        log.err("IREE parse failed (see diagnostics above)", .{});
        return error.ParseFailed;
    }

    if (!vt.invocation_pipeline(inv, @intFromEnum(PipelineType.std))) {
        log.err("IREE pipeline (std) failed", .{});
        return error.PipelineFailed;
    }

    var output: ?*Output = null;
    if (vt.output_open_membuffer(&output)) |err| {
        defer vt.error_destroy(err);
        const msg = vt.error_get_message(err);
        log.err("IREE output_open_membuffer failed: {s}", .{msg});
        return error.OutputOpenFailed;
    }
    defer vt.output_destroy(output);

    if (vt.invocation_output_vm_bytecode(inv, output)) |err| {
        defer vt.error_destroy(err);
        const msg = vt.error_get_message(err);
        log.err("IREE output_vm_bytecode failed: {s}", .{msg});
        return error.OutputFailed;
    }

    var data_ptr: ?*anyopaque = null;
    var data_len: u64 = 0;
    if (vt.output_map_memory(output, &data_ptr, &data_len)) |err| {
        defer vt.error_destroy(err);
        const msg = vt.error_get_message(err);
        log.err("IREE output_map_memory failed: {s}", .{msg});
        return error.MapMemoryFailed;
    }

    const src_bytes: []const u8 = @as([*]const u8, @ptrCast(data_ptr.?))[0..data_len];
    const result = try allocator.dupe(u8, src_bytes);

    vt.output_keep(output);

    log.debug("compile: {d} bytes MLIR -> {d} bytes VMFB", .{ mlir_bytes.len, result.len });
    return result;
}

// ---------------------------------------------------------------------------
// Internal: diagnostic callback forwarded to std.log.
// ---------------------------------------------------------------------------

/// Diagnostic context threaded through the IREE compiler callback.
///
/// Currently unused by the handler (diagnostics go straight to std.log),
/// but kept so callers can later collect errors programmatically.
const DiagCtx = struct {
    vt: *const Vtable,
};

fn diag_handler(
    severity: DiagnosticSeverity,
    message: [*c]const u8,
    message_size: usize,
    user_data: ?*anyopaque,
) callconv(.c) void {
    _ = user_data;
    const msg = message[0..message_size];
    switch (severity) {
        .note, .remark => log.debug("{s}", .{msg}),
        .warning => log.debug("warning: {s}", .{msg}),
        .err => log.err("{s}", .{msg}),
    }
}
