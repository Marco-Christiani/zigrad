/// IREE Compiler Embedding API bindings.
///
/// Loads `libIREECompiler.so` at runtime via dlopen and resolves all embedding
/// API symbols as function pointers.  No build-time link against libIREECompiler
/// is required -- the library is entirely optional and loaded on demand.
///
/// ## Usage
///
/// ```zig
/// var cmp = try Compiler.load("/path/to/libIREECompiler.so");
/// defer cmp.unload();
/// cmp.global_init();
/// defer cmp.global_shutdown();
///
/// const vmfb = try cmp.compile(allocator, mlir_bytes, is_bytecode);
/// defer allocator.free(vmfb);
/// ```
///
/// ## Reference
///
/// IREE embedding API: `iree/compiler/embedding_api.h`
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

pub const DiagnosticCallback = *const fn (DiagnosticSeverity, [*c]const u8, ?*anyopaque) callconv(.c) void;

// ---------------------------------------------------------------------------
// Function pointer table (resolved via dlsym after dlopen).
// ---------------------------------------------------------------------------

const Vtable = struct {
    get_api_version: *const fn () callconv(.c) c_int,
    global_initialize: *const fn () callconv(.c) void,
    global_shutdown: *const fn () callconv(.c) void,

    session_create: *const fn () callconv(.c) ?*Session,
    session_destroy: *const fn (?*Session) callconv(.c) void,
    session_set_flags: *const fn (?*Session, c_int, [*c]const [*c]const u8) callconv(.c) void,
    session_set_diagnostic_handler: *const fn (?*Session, DiagnosticCallback, ?*anyopaque) callconv(.c) void,

    invocation_create: *const fn (?*Session) callconv(.c) ?*Invocation,
    invocation_destroy: *const fn (?*Invocation) callconv(.c) void,
    invocation_parse_source: *const fn (?*Invocation, ?*Source) callconv(.c) ?*Error,
    invocation_pipeline: *const fn (?*Invocation, c_int) callconv(.c) bool,
    invocation_output_vm_bytecode: *const fn (?*Invocation, ?*Output) callconv(.c) bool,

    source_wrap_buffer: *const fn (?*Session, [*c]const u8, [*c]const u8, usize, bool) callconv(.c) ?*Source,
    source_destroy: *const fn (?*Source) callconv(.c) void,

    output_open_membuffer: *const fn () callconv(.c) ?*Output,
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

/// Handle to a loaded libIREECompiler.so.
///
/// Call `global_init()` after loading and `global_shutdown()` before unloading.
/// The library is kept loaded for process lifetime (IREE compiler has global
/// state that is not safe to unload and reload).
pub const Compiler = struct {
    lib: std.DynLib,
    vt: Vtable,

    /// Load `libIREECompiler.so` from `path`.
    ///
    /// Does NOT call `ireeCompilerGlobalInitialize` -- call `global_init()` separately.
    pub fn load(path: []const u8) !Compiler {
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
        const major = ver >> 16;
        const minor = ver & 0xFFFF;
        log.info("IREE compiler API version: {d}.{d}", .{ major, minor });

        return .{ .lib = lib, .vt = vt };
    }

    fn load_vtable(lib: *std.DynLib) !Vtable {
        return .{
            .get_api_version = try resolve(@TypeOf(@as(Vtable, undefined).get_api_version), lib, "ireeCompilerGetAPIVersion"),
            .global_initialize = try resolve(@TypeOf(@as(Vtable, undefined).global_initialize), lib, "ireeCompilerGlobalInitialize"),
            .global_shutdown = try resolve(@TypeOf(@as(Vtable, undefined).global_shutdown), lib, "ireeCompilerGlobalShutdown"),
            .session_create = try resolve(@TypeOf(@as(Vtable, undefined).session_create), lib, "ireeCompilerSessionCreate"),
            .session_destroy = try resolve(@TypeOf(@as(Vtable, undefined).session_destroy), lib, "ireeCompilerSessionDestroy"),
            .session_set_flags = try resolve(@TypeOf(@as(Vtable, undefined).session_set_flags), lib, "ireeCompilerSessionSetFlags"),
            .session_set_diagnostic_handler = try resolve(@TypeOf(@as(Vtable, undefined).session_set_diagnostic_handler), lib, "ireeCompilerSessionSetDiagnosticHandler"),
            .invocation_create = try resolve(@TypeOf(@as(Vtable, undefined).invocation_create), lib, "ireeCompilerInvocationCreate"),
            .invocation_destroy = try resolve(@TypeOf(@as(Vtable, undefined).invocation_destroy), lib, "ireeCompilerInvocationDestroy"),
            .invocation_parse_source = try resolve(@TypeOf(@as(Vtable, undefined).invocation_parse_source), lib, "ireeCompilerInvocationParseSource"),
            .invocation_pipeline = try resolve(@TypeOf(@as(Vtable, undefined).invocation_pipeline), lib, "ireeCompilerInvocationPipeline"),
            .invocation_output_vm_bytecode = try resolve(@TypeOf(@as(Vtable, undefined).invocation_output_vm_bytecode), lib, "ireeCompilerInvocationOutputVMBytecode"),
            .source_wrap_buffer = try resolve(@TypeOf(@as(Vtable, undefined).source_wrap_buffer), lib, "ireeCompilerSourceWrapBuffer"),
            .source_destroy = try resolve(@TypeOf(@as(Vtable, undefined).source_destroy), lib, "ireeCompilerSourceDestroy"),
            .output_open_membuffer = try resolve(@TypeOf(@as(Vtable, undefined).output_open_membuffer), lib, "ireeCompilerOutputOpenMembuffer"),
            .output_destroy = try resolve(@TypeOf(@as(Vtable, undefined).output_destroy), lib, "ireeCompilerOutputDestroy"),
            .output_map_memory = try resolve(@TypeOf(@as(Vtable, undefined).output_map_memory), lib, "ireeCompilerOutputMapMemory"),
            .output_keep = try resolve(@TypeOf(@as(Vtable, undefined).output_keep), lib, "ireeCompilerOutputKeep"),
            .error_destroy = try resolve(@TypeOf(@as(Vtable, undefined).error_destroy), lib, "ireeCompilerErrorDestroy"),
            .error_get_message = try resolve(@TypeOf(@as(Vtable, undefined).error_get_message), lib, "ireeCompilerErrorGetMessage"),
        };
    }

    pub fn unload(self: *Compiler) void {
        self.lib.close();
    }

    pub fn global_init(self: *const Compiler) void {
        self.vt.global_initialize();
    }

    pub fn global_shutdown(self: *const Compiler) void {
        self.vt.global_shutdown();
    }

    /// Compile `mlir_bytes` to VMFB (VM FlatBuffer) bytes.
    ///
    /// Caller owns the returned slice; free with `allocator.free`.
    ///
    /// `is_bytecode`: true if `mlir_bytes` contains MLIR bytecode (binary),
    ///   false if it contains MLIR text.
    ///
    /// Returns error if the compiler session/invocation fails or if memory
    /// allocation fails.
    pub fn compile(
        self: *const Compiler,
        allocator: std.mem.Allocator,
        mlir_bytes: []const u8,
        is_bytecode: bool,
    ) ![]u8 {
        const session = self.vt.session_create() orelse return error.SessionCreateFailed;
        defer self.vt.session_destroy(session);

        // Install diagnostic handler that routes to our logger.
        // DiagCtx is passed as user_data for future use (e.g. collecting
        // errors to return from compile() instead of only logging).
        var diag_ctx: DiagCtx = .{ .vt = &self.vt };
        self.vt.session_set_diagnostic_handler(session, diag_handler, @ptrCast(&diag_ctx));

        const inv = self.vt.invocation_create(session) orelse return error.InvocationCreateFailed;
        defer self.vt.invocation_destroy(inv);

        // Wrap the input MLIR buffer (bytecode or text).
        // For bytecode, isNullTerminated=false; for text, IREE expects null-term.
        const source = self.vt.source_wrap_buffer(
            session,
            "input",
            mlir_bytes.ptr,
            mlir_bytes.len,
            !is_bytecode, // null-terminated only needed for text
        ) orelse return error.SourceWrapFailed;
        defer self.vt.source_destroy(source);

        if (self.vt.invocation_parse_source(inv, source)) |err| {
            defer self.vt.error_destroy(err);
            const msg = self.vt.error_get_message(err);
            log.err("IREE parse failed: {s}", .{msg});
            return error.ParseFailed;
        }

        if (!self.vt.invocation_pipeline(inv, @intFromEnum(PipelineType.std))) {
            log.err("IREE pipeline (std) failed", .{});
            return error.PipelineFailed;
        }

        const output = self.vt.output_open_membuffer() orelse return error.OutputOpenFailed;
        defer self.vt.output_destroy(output);

        if (!self.vt.invocation_output_vm_bytecode(inv, output)) {
            log.err("IREE output_vm_bytecode failed", .{});
            return error.OutputFailed;
        }

        var data_ptr: ?*anyopaque = null;
        var data_len: u64 = 0;
        if (self.vt.output_map_memory(output, &data_ptr, &data_len)) |err| {
            defer self.vt.error_destroy(err);
            const msg = self.vt.error_get_message(err);
            log.err("IREE output_map_memory failed: {s}", .{msg});
            return error.MapMemoryFailed;
        }

        const src_bytes: []const u8 = @as([*]const u8, @ptrCast(data_ptr.?))[0..data_len];
        const result = try allocator.dupe(u8, src_bytes);

        // Keep the membuffer alive until we've copied out; then let defer destroy it.
        self.vt.output_keep(output);

        log.debug("compile: {d} bytes MLIR -> {d} bytes VMFB", .{ mlir_bytes.len, result.len });
        return result;
    }
};

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
    user_data: ?*anyopaque,
) callconv(.c) void {
    _ = user_data;
    const msg = std.mem.span(message);
    switch (severity) {
        .note, .remark => log.debug("{s}", .{msg}),
        .warning => log.debug("warning: {s}", .{msg}),
        .err => log.err("{s}", .{msg}),
    }
}
