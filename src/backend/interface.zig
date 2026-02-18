/// Backend interface conformance checker.
///
/// `AsBackend(Module)` validates at compile time that `Module` (the backend's
/// top-level namespace) exports a `Backend` struct with the right methods AND
/// the required associated handle types. Returns `Module.Backend`.
///
/// Associated types (`Buffer`, `Device`, etc.) live at module scope — NOT inside
/// the `Backend` struct — because Zig 0.15 treats same-name decls in both scopes
/// as ambiguous within method signatures.
///
/// This does NOT enforce exact error sets (Zig infers those per-backend).
/// It checks: param count, self type (`*Backend`), and return category (void vs
/// error union with payload type).
const std = @import("std");
const pr = @import("../pr/pr.zig");

pub fn AsBackend(comptime Module: type) type {
    // -- Backend struct must exist ------------------------------------------
    if (!@hasDecl(Module, "Backend")) {
        @compileError(std.fmt.comptimePrint(
            "Backend module {s} missing 'Backend' struct",
            .{@typeName(Module)},
        ));
    }
    const T = Module.Backend;

    // -- Associated types (module-level) ------------------------------------
    require_type(Module, "Buffer");
    require_type(Module, "RawBuffer");
    require_type(Module, "Event");
    require_type(Module, "Device");
    require_type(Module, "LoadedExecutable");
    require_type(Module, "ExecuteResult");
    require_type(Module, "CompileOptions");

    const Buffer = @field(Module, "Buffer");
    const RawBuffer = @field(Module, "RawBuffer");
    const Event = @field(Module, "Event");
    const Device = @field(Module, "Device");
    const LoadedExecutable = @field(Module, "LoadedExecutable");
    const ExecuteResult = @field(Module, "ExecuteResult");
    const CompileOptions = @field(Module, "CompileOptions");

    // -- Lifecycle ---------------------------------------------------------
    check_method(T, "deinit", &.{*T}, void);
    check_method(T, "get_devices", &.{ *T, std.mem.Allocator }, []Device);

    // -- Compilation -------------------------------------------------------
    check_method(T, "compile", &.{ *T, *const Device, []const u8, bool, CompileOptions }, LoadedExecutable);

    // -- Buffer management -------------------------------------------------
    check_method(T, "buffer_from_host", &.{ *T, *const Device, []const u8, pr.DType, []const i64 }, Buffer);

    // -- Execution ---------------------------------------------------------
    check_method(T, "execute", &.{ *T, *LoadedExecutable, std.mem.Allocator, []const Buffer }, ExecuteResult);
    check_method(T, "execute_into", &.{ *T, *LoadedExecutable, []const RawBuffer, []?RawBuffer, ?[]const i64 }, ?Event);

    // -- Handle lifecycle --------------------------------------------------
    check_method(T, "deinit_buffer", &.{ *T, *Buffer }, void);
    check_method(T, "deinit_event", &.{ *T, *Event }, void);
    check_method(T, "deinit_executable", &.{ *T, *LoadedExecutable }, void);

    // -- Data transfer -----------------------------------------------------
    check_method(T, "buffer_to_host", &.{ *T, *Buffer, []u8 }, Event);
    check_method(T, "await_event", &.{ *T, *Event }, void);

    return T;
}

fn require_type(comptime Ns: type, comptime name: []const u8) void {
    if (!@hasDecl(Ns, name)) {
        @compileError(std.fmt.comptimePrint(
            "Backend module {s} missing associated type '{s}'",
            .{ @typeName(Ns), name },
        ));
    }
}

fn check_method(
    comptime T: type,
    comptime name: []const u8,
    comptime expected_params: []const type,
    comptime expected_payload: type,
) void {
    if (!@hasDecl(T, name)) {
        @compileError(std.fmt.comptimePrint(
            "Backend {s} missing method '{s}'",
            .{ @typeName(T), name },
        ));
    }
    const func = @field(T, name);
    const info = @typeInfo(@TypeOf(func));
    if (info != .@"fn") {
        @compileError(std.fmt.comptimePrint(
            "Backend {s}.{s} is not a function",
            .{ @typeName(T), name },
        ));
    }
    const fn_info = info.@"fn";

    // Check param count
    if (fn_info.params.len != expected_params.len) {
        @compileError(std.fmt.comptimePrint(
            "Backend {s}.{s}: expected {d} params, got {d}",
            .{ @typeName(T), name, expected_params.len, fn_info.params.len },
        ));
    }

    // Check self type (first param)
    if (expected_params.len > 0) {
        const self_type = fn_info.params[0].type orelse @compileError(std.fmt.comptimePrint(
            "Backend {s}.{s}: self param type is generic",
            .{ @typeName(T), name },
        ));
        if (self_type != expected_params[0]) {
            @compileError(std.fmt.comptimePrint(
                "Backend {s}.{s}: self param should be {s}, got {s}",
                .{ @typeName(T), name, @typeName(expected_params[0]), @typeName(self_type) },
            ));
        }
    }

    // Check return: void vs error union with payload
    const ret = fn_info.return_type orelse @compileError(std.fmt.comptimePrint(
        "Backend {s}.{s}: return type is generic",
        .{ @typeName(T), name },
    ));

    const actual_payload = switch (@typeInfo(ret)) {
        .error_union => |eu| eu.payload,
        else => ret,
    };

    if (expected_payload == void) {
        if (actual_payload != void) {
            @compileError(std.fmt.comptimePrint(
                "Backend {s}.{s}: expected void return, got {s}",
                .{ @typeName(T), name, @typeName(ret) },
            ));
        }
    } else {
        if (actual_payload != expected_payload) {
            @compileError(std.fmt.comptimePrint(
                "Backend {s}.{s}: expected return payload {s}, got {s}",
                .{ @typeName(T), name, @typeName(expected_payload), @typeName(actual_payload) },
            ));
        }
    }
}

test "pjrt backend conforms" {
    _ = AsBackend(@import("pjrt.zig"));
}
