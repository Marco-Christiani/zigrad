/// M4.2 Milestone Test: Custom Call Boundaries with Typed FFI
///
/// Tests custom call execution with handler registration via PJRT FFI extension.
/// Runs in two modes:
/// - --no-handler: Negative test (should fail without registered handler)
/// - --with-handler: Positive test (should succeed with registered handler)
///
const std = @import("std");
const zigrad = @import("zigrad");
const mlir = @import("mlir/mlir.zig");
const stablehlo = @import("mlir/dialects/stablehlo.zig");
const term_color = @import("util/term_color.zig");
const pjrt_plugin = zigrad.pjrt.plugin;
const pjrt_api = zigrad.pjrt.api;
const c_mod = zigrad.pjrt.c;
const c = c_mod.c;

const Backend = zigrad.Backend;
const HostBuffer = zigrad.HostBuffer;
const Shape = zigrad.Shape;
const Program = zigrad.Program;

// Custom call target name
const CALL_TARGET_NAME = "zg_custom_zero";

const CudaMemsetAsyncFn = *const fn (?*anyopaque, c_int, usize, ?*anyopaque) callconv(.c) c_int;

fn makeFfiError(frame: *const c.XLA_FFI_CallFrame, code: c.XLA_FFI_Error_Code, msg: []const u8) ?*c.XLA_FFI_Error {
    if (frame.api == null) return null;
    const create_fn = frame.api.*.XLA_FFI_Error_Create orelse return null;
    var args = std.mem.zeroes(c.XLA_FFI_Error_Create_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Error_Create_Args);
    args.extension_start = null;
    args.message = msg.ptr;
    args.errc = code;
    return create_fn(&args);
}

fn getCudaMemsetAsync() ?CudaMemsetAsyncFn {
    const sym = c.dlsym(c.RTLD_DEFAULT, "cudaMemsetAsync") orelse return null;
    return @ptrCast(sym);
}

fn getCudaStream(frame: *const c.XLA_FFI_CallFrame) ?*anyopaque {
    if (frame.api == null) return null;
    const get_fn = frame.api.*.XLA_FFI_Stream_Get orelse return null;

    var args = std.mem.zeroes(c.XLA_FFI_Stream_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Stream_Get_Args);
    args.extension_start = null;
    args.ctx = frame.ctx;
    args.stream = null;

    if (get_fn(&args) != null) return null;
    return args.stream;
}

fn maybeFillMetadata(frame: *const c.XLA_FFI_CallFrame) bool {
    var ext: ?*c.XLA_FFI_Extension_Base = frame.extension_start;
    while (ext) |e| {
        if (e.type == c.XLA_FFI_Extension_Metadata) {
            const meta_ext: *c.XLA_FFI_Metadata_Extension = @ptrCast(@alignCast(e));
            if (meta_ext.metadata) |meta| {
                const meta_ptr: *c.XLA_FFI_Metadata = @ptrCast(@alignCast(meta));
                meta_ptr.*.api_version.struct_size = @sizeOf(c.XLA_FFI_Api_Version);
                meta_ptr.*.api_version.extension_start = null;
                meta_ptr.*.api_version.major_version = c.XLA_FFI_API_MAJOR;
                meta_ptr.*.api_version.minor_version = c.XLA_FFI_API_MINOR;
                meta_ptr.*.traits = 0;
                meta_ptr.*.state_type_id = .{ .type_id = 0 };
            }
            return true;
        }
        ext = e.next;
    }
    return false;
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

fn logApiPointers(api: *const c.PJRT_Api, ffi_ext: ?*c.PJRT_FFI_Extension) void {
    if (!debugEnabled()) return;
    const ext_base = api.extension_start;
    std.debug.print("[pjrt-debug] PJRT_Api ptr={*} extension_start={*}\n", .{ api, ext_base });
    if (ext_base) |ptr| {
        logDladdr("PJRT_Api extension_start", @ptrCast(@constCast(ptr)));
    }
    if (ffi_ext) |ffi| {
        std.debug.print("[pjrt-debug] PJRT_FFI_Extension ptr={*} register_handler={*}\n", .{
            ffi,
            ffi.register_handler,
        });
        if (ffi.register_handler) |reg_fn| {
            logDladdr("PJRT_FFI_Extension.register_handler", @ptrCast(@constCast(reg_fn)));
        }
    }
}

//
// MINIMAL TYPED-FFI HANDLER
//
// This handler zeroes all output buffers, demonstrating the minimal working pattern.
//
export fn zg_custom_zero(call_frame: *c.XLA_FFI_CallFrame) callconv(.c) ?*c.XLA_FFI_Error {
    const frame = call_frame.*;

    if (maybeFillMetadata(&frame)) {
        return null;
    }

    const memset_async = getCudaMemsetAsync() orelse {
        return makeFfiError(&frame, c.XLA_FFI_Error_Code_INTERNAL, "cudaMemsetAsync not available");
    };
    const stream = getCudaStream(&frame) orelse {
        return makeFfiError(&frame, c.XLA_FFI_Error_Code_INTERNAL, "XLA FFI stream not available");
    };

    // Zero-fill all output buffers on the device stream.
    const num_rets: usize = @intCast(frame.rets.size);
    var i: usize = 0;
    while (i < num_rets) : (i += 1) {
        const buf_ptr: *c.XLA_FFI_Buffer = @ptrCast(@alignCast(frame.rets.rets[i]));
        const buf = buf_ptr.*;

        var size: usize = 1;
        var dim_idx: usize = 0;
        while (dim_idx < buf.rank) : (dim_idx += 1) {
            size *= @intCast(buf.dims[dim_idx]);
        }

        const elem_size: usize = switch (buf.dtype) {
            c.XLA_FFI_DataType_F32 => 4,
            c.XLA_FFI_DataType_F64 => 8,
            c.XLA_FFI_DataType_S32 => 4,
            c.XLA_FFI_DataType_S64 => 8,
            else => 4,
        };

        const total_bytes = size * elem_size;
        const rc = memset_async(buf.data, 0, total_bytes, stream);
        if (rc != 0) {
            return makeFfiError(&frame, c.XLA_FFI_Error_Code_INTERNAL, "cudaMemsetAsync failed");
        }
    }

    return null;
}

/// Walk PJRT extension chain to find FFI extension
fn findFfiExtension(api: *const c.PJRT_Api) ?*c.PJRT_FFI_Extension {
    var ext: ?*c.PJRT_Extension_Base = api.extension_start;

    while (ext) |e| {
        if (e.type == c.PJRT_Extension_Type_FFI) {
            return @ptrCast(@alignCast(e));
        }
        ext = e.next;
    }

    return null;
}

fn findGpuCustomCallExtension(api: *const c.PJRT_Api) ?*c.PJRT_Gpu_Custom_Call {
    var ext: ?*c.PJRT_Extension_Base = api.extension_start;

    while (ext) |e| {
        if (e.type == c.PJRT_Extension_Type_Gpu_Custom_Call) {
            return @ptrCast(@alignCast(e));
        }
        ext = e.next;
    }

    return null;
}

/// Register custom call handler via PJRT FFI extension
fn registerHandler(api: *const c.PJRT_Api, platform: []const u8, size_variant: u8) !void {
    if (findGpuCustomCallExtension(api)) |gpu_ext| {
        var args = pjrt_api.initArgs(c.PJRT_Gpu_Register_Custom_Call_Args);
        args.function_name = CALL_TARGET_NAME.ptr;
        args.function_name_size = CALL_TARGET_NAME.len;
        args.api_version = 1;
        args.handler_instantiate = null;
        args.handler_prepare = null;
        args.handler_initialize = null;
        args.handler_execute = @ptrCast(@constCast(&zg_custom_zero));

        if (gpu_ext.custom_call) |reg_fn| {
            if (reg_fn(&args)) |pjrt_err| {
                std.debug.print("ERROR: GPU custom call registration failed\n", .{});
                var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
                @field(msg_args, "error") = pjrt_err;
                if (api.PJRT_Error_Message) |msg_fn| {
                    _ = msg_fn(&msg_args);
                    if (msg_args.message) |msg| {
                        const message = msg[0..msg_args.message_size];
                        std.debug.print("  Error message: {s}\n", .{message});
                    }
                }
                return error.HandlerRegistrationFailed;
            }

            std.debug.print("Successfully registered handler '{s}' via GPU custom call extension\n", .{CALL_TARGET_NAME});
            return;
        }
    }

    const ffi_ext = findFfiExtension(api) orelse {
        std.debug.print("ERROR: PJRT_Extension_Type_FFI not found\n", .{});
        return error.FfiExtensionNotFound;
    };

    logApiPointers(api, ffi_ext);

    std.debug.print("Found PJRT FFI extension\n", .{});

    // Compute sizes based on variant
    const target_size: usize = switch (size_variant) {
        0 => CALL_TARGET_NAME.len, // Variant A: exclude NUL
        1 => CALL_TARGET_NAME.len + 1, // Variant B: include NUL
        2 => 0, // Variant C: let runtime compute
        else => CALL_TARGET_NAME.len,
    };
    const platform_size: usize = switch (size_variant) {
        0 => platform.len, // Variant A: exclude NUL
        1 => platform.len + 1, // Variant B: include NUL
        2 => 0, // Variant C: let runtime compute
        else => platform.len,
    };

    // Prepare registration args - pass raw function pointer as XLA_FFI_Handler*
    var register_args = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
    register_args.target_name = CALL_TARGET_NAME.ptr;
    register_args.target_name_size = target_size;
    register_args.handler = @ptrCast(@constCast(&zg_custom_zero)); // Raw function pointer
    register_args.platform_name = platform.ptr;
    register_args.platform_name_size = platform_size;
    register_args.traits = 0; // No special traits

    // DEBUG: Print exact registration parameters
    std.debug.print("=== REGISTRATION DEBUG (Variant {}) ===\n", .{size_variant});
    std.debug.print("  target_name: '{s}' (ptr={*})\n", .{ CALL_TARGET_NAME, CALL_TARGET_NAME.ptr });
    std.debug.print("  target_name_size: {} (computed len={})\n", .{ register_args.target_name_size, CALL_TARGET_NAME.len });
    std.debug.print("  platform_name: '{s}' (ptr={*})\n", .{ platform, platform.ptr });
    std.debug.print("  platform_name_size: {} (computed len={})\n", .{ register_args.platform_name_size, platform.len });
    std.debug.print("  handler: {*}\n", .{register_args.handler});
    std.debug.print("  traits: 0x{x}\n", .{register_args.traits});
    std.debug.print("  struct_size: {} (expected={})\n", .{ register_args.struct_size, @sizeOf(c.PJRT_FFI_Register_Handler_Args) });
    std.debug.print("=====================================\n", .{});

    // Register handler
    const register_fn = ffi_ext.register_handler orelse {
        std.debug.print("ERROR: register_handler function pointer is null\n", .{});
        return error.RegisterHandlerNotAvailable;
    };

    if (register_fn(&register_args)) |pjrt_err| {
        std.debug.print("ERROR: Handler registration failed\n", .{});

        // Try to get error message
        var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
        @field(msg_args, "error") = pjrt_err;
        if (api.PJRT_Error_Message) |msg_fn| {
            _ = msg_fn(&msg_args);
            if (msg_args.message) |msg| {
                const message = msg[0..msg_args.message_size];
                std.debug.print("  Error message: {s}\n", .{message});
            }
        }

        return error.HandlerRegistrationFailed;
    }

    std.debug.print("Successfully registered handler '{s}' for platform '{s}'\n", .{ CALL_TARGET_NAME, platform });
}

/// Test registration API with invalid arguments to validate it's actually functioning
fn testRegistrationSanity(api: *const c.PJRT_Api) !void {
    const ffi_ext = findFfiExtension(api) orelse {
        std.debug.print("ERROR: PJRT_Extension_Type_FFI not found\n", .{});
        return error.FfiExtensionNotFound;
    };

    const register_fn = ffi_ext.register_handler orelse {
        std.debug.print("ERROR: register_handler function pointer is null\n", .{});
        return error.RegisterHandlerNotAvailable;
    };

    std.debug.print("\n", .{});
    std.debug.print("╔════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║  FFI REGISTRATION SANITY CHECKS                        ║\n", .{});
    std.debug.print("╚════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});

    // Test 1: Invalid platform name
    {
        std.debug.print("[Test 1] Invalid platform name: 'not_a_platform'\n", .{});
        const invalid_platform = "not_a_platform";
        var register_args = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
        register_args.target_name = CALL_TARGET_NAME.ptr;
        register_args.target_name_size = CALL_TARGET_NAME.len;
        register_args.handler = @ptrCast(@constCast(&zg_custom_zero));
        register_args.platform_name = invalid_platform.ptr;
        register_args.platform_name_size = invalid_platform.len;
        register_args.traits = 0;

        if (register_fn(&register_args)) |pjrt_err| {
            std.debug.print("  ✓ Registration REJECTED (as expected)\n", .{});
            var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
            @field(msg_args, "error") = pjrt_err;
            if (api.PJRT_Error_Message) |msg_fn| {
                _ = msg_fn(&msg_args);
                if (msg_args.message) |msg| {
                    const message = msg[0..msg_args.message_size];
                    std.debug.print("    Error: {s}\n", .{message});
                }
            }
        } else {
            std.debug.print("  ⚠ Registration SUCCEEDED (unexpected!)\n", .{});
        }
        std.debug.print("\n", .{});
    }

    // Test 2: Null handler pointer
    {
        std.debug.print("[Test 2] Null handler pointer\n", .{});
        var register_args = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
        register_args.target_name = CALL_TARGET_NAME.ptr;
        register_args.target_name_size = CALL_TARGET_NAME.len;
        register_args.handler = null; // NULL handler
        register_args.platform_name = "cuda".ptr;
        register_args.platform_name_size = 4;
        register_args.traits = 0;

        if (register_fn(&register_args)) |pjrt_err| {
            std.debug.print("  ✓ Registration REJECTED (as expected)\n", .{});
            var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
            @field(msg_args, "error") = pjrt_err;
            if (api.PJRT_Error_Message) |msg_fn| {
                _ = msg_fn(&msg_args);
                if (msg_args.message) |msg| {
                    const message = msg[0..msg_args.message_size];
                    std.debug.print("    Error: {s}\n", .{message});
                }
            }
        } else {
            std.debug.print("  ⚠ Registration SUCCEEDED (unexpected!)\n", .{});
        }
        std.debug.print("\n", .{});
    }

    // Test 3: Empty target name
    {
        std.debug.print("[Test 3] Empty target name (size=0)\n", .{});
        var register_args = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
        register_args.target_name = CALL_TARGET_NAME.ptr; // Non-null but size 0
        register_args.target_name_size = 0; // Empty
        register_args.handler = @ptrCast(@constCast(&zg_custom_zero));
        register_args.platform_name = "cuda".ptr;
        register_args.platform_name_size = 4;
        register_args.traits = 0;

        if (register_fn(&register_args)) |pjrt_err| {
            std.debug.print("  ✓ Registration REJECTED (as expected)\n", .{});
            var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
            @field(msg_args, "error") = pjrt_err;
            if (api.PJRT_Error_Message) |msg_fn| {
                _ = msg_fn(&msg_args);
                if (msg_args.message) |msg| {
                    const message = msg[0..msg_args.message_size];
                    std.debug.print("    Error: {s}\n", .{message});
                }
            }
        } else {
            std.debug.print("  ⚠ Registration SUCCEEDED (unexpected!)\n", .{});
        }
        std.debug.print("\n", .{});
    }

    // Test 4: Duplicate registration
    {
        std.debug.print("[Test 4] Duplicate registration (register same handler twice)\n", .{});

        // First registration
        std.debug.print("  First registration...\n", .{});
        var register_args1 = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
        register_args1.target_name = CALL_TARGET_NAME.ptr;
        register_args1.target_name_size = CALL_TARGET_NAME.len;
        register_args1.handler = @ptrCast(@constCast(&zg_custom_zero));
        register_args1.platform_name = "cuda".ptr;
        register_args1.platform_name_size = 4;
        register_args1.traits = 0;

        if (register_fn(&register_args1)) |pjrt_err| {
            std.debug.print("    ✗ First registration FAILED\n", .{});
            var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
            @field(msg_args, "error") = pjrt_err;
            if (api.PJRT_Error_Message) |msg_fn| {
                _ = msg_fn(&msg_args);
                if (msg_args.message) |msg| {
                    const message = msg[0..msg_args.message_size];
                    std.debug.print("      Error: {s}\n", .{message});
                }
            }
        } else {
            std.debug.print("    ✓ First registration succeeded\n", .{});

            // Second registration (duplicate)
            std.debug.print("  Second registration (duplicate)...\n", .{});
            var register_args2 = pjrt_api.initArgs(c.PJRT_FFI_Register_Handler_Args);
            register_args2.target_name = CALL_TARGET_NAME.ptr;
            register_args2.target_name_size = CALL_TARGET_NAME.len;
            register_args2.handler = @ptrCast(@constCast(&zg_custom_zero));
            register_args2.platform_name = "cuda".ptr;
            register_args2.platform_name_size = 4;
            register_args2.traits = 0;

            if (register_fn(&register_args2)) |pjrt_err| {
                std.debug.print("    ✓ Duplicate registration REJECTED (as expected)\n", .{});
                var msg_args = pjrt_api.initArgs(c.PJRT_Error_Message_Args);
                @field(msg_args, "error") = pjrt_err;
                if (api.PJRT_Error_Message) |msg_fn| {
                    _ = msg_fn(&msg_args);
                    if (msg_args.message) |msg| {
                        const message = msg[0..msg_args.message_size];
                        std.debug.print("      Error: {s}\n", .{message});
                    }
                }
            } else {
                std.debug.print("    ⚠ Duplicate registration SUCCEEDED (unexpected - no uniqueness check?)\n", .{});
            }
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("╔════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║  SANITY CHECKS COMPLETE                                ║\n", .{});
    std.debug.print("╚════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
}

/// Build MLIR module with custom_call using typed FFI
fn buildModule(mlir_ctx: mlir.Context, allocator: std.mem.Allocator) ![]const u8 {
    const loc = mlir.Location.unknown(mlir_ctx);

    // Create module
    var module = mlir.Module.init(loc);
    defer module.deinit();

    // Build function type: (tensor<2x3xf32>) -> tensor<2x3xf32>
    const f32_type = mlir.Type.float(mlir_ctx, .f32);
    const input_type = mlir.Type.tensor(&.{ 2, 3 }, f32_type);
    const output_type = mlir.Type.tensor(&.{ 2, 3 }, f32_type);

    const func_type = mlir.Type.function(mlir_ctx, &.{input_type}, &.{output_type});

    // Create function body block
    const entry_block = try mlir.Block.init(&.{input_type}, &.{loc});

    const arg0_val = entry_block.argument(0);

    // Build stablehlo.custom_call with TYPED_FFI api_version and empty backend_config dict.
    const custom_call_op = stablehlo.custom_call(mlir_ctx, &.{arg0_val}, .{
        .call_target_name = CALL_TARGET_NAME,
        .has_side_effect = false,
        .backend_config = mlir.Attribute.dict(mlir_ctx, &.{}),
        .api_version = .typed_ffi,
    }, &.{output_type}, loc);
    entry_block.appendOperation(custom_call_op);

    // DEBUG: Print custom_call emission details
    std.debug.print("=== CUSTOM_CALL EMISSION DEBUG ===\n", .{});
    std.debug.print("  call_target_name: '{s}'\n", .{CALL_TARGET_NAME});
    std.debug.print("  call_target_name length: {}\n", .{CALL_TARGET_NAME.len});
    std.debug.print("  api_version: 4 (TYPED_FFI)\n", .{});
    std.debug.print("===================================\n", .{});

    // Build return operation
    const return_op = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{custom_call_op.result(0)},
        .verify = false,
        .location = loc,
    });
    entry_block.appendOperation(return_op);

    // Create function operation
    const func_op = mlir.Operation.make(mlir_ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, "main") },
            .{ "function_type", mlir.Attribute.type_(func_type) },
        },
        .location = loc,
    });

    module.getBody().appendOperation(func_op);

    // Verify module
    if (!module.op().verify()) {
        return error.InvalidMlir;
    }

    // Serialize to bytecode
    var bytecode_buffer_fixed: [1024 * 1024]u8 = undefined;
    var bytecode_writer: std.Io.Writer = .fixed(&bytecode_buffer_fixed);
    try module.op().writeBytecode(&bytecode_writer);

    return try allocator.dupe(u8, bytecode_writer.buffered());
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [2048]u8 = undefined;
    var stdout_writer = std.fs.File.stderr().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch |e| switch (e) {
        error.WriteFailed => @panic("write failed on flush"),
    };

    var tty = term_color.Tty.initForStderr(stdout);

    // Parse arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var with_handler = false;
    var size_variant: u8 = 0; // Default: exclude NUL (Variant A)
    var ffi_reg_sanity = false;

    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "--with-handler")) {
            with_handler = true;
        } else if (std.mem.eql(u8, args[1], "--no-handler")) {
            with_handler = false;
        } else {
            try stdout.print("Usage: {s} [--with-handler|--no-handler] [--size-variant=0|1|2] [--ffi-reg-sanity]\n", .{args[0]});
            try stdout.print("  Size variants: 0=exclude NUL (default), 1=include NUL, 2=size=0\n", .{});
            try stdout.print("  --ffi-reg-sanity: Run registration API validation experiments\n", .{});
            return error.InvalidArguments;
        }
    }

    // Optional parameters
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.startsWith(u8, args[i], "--size-variant=")) {
            const variant_str = args[i]["--size-variant=".len..];
            size_variant = std.fmt.parseInt(u8, variant_str, 10) catch {
                try stdout.print("Invalid size variant: {s}\n", .{variant_str});
                return error.InvalidArguments;
            };
            if (size_variant > 2) {
                try stdout.print("Size variant must be 0, 1, or 2\n", .{});
                return error.InvalidArguments;
            }
        } else if (std.mem.eql(u8, args[i], "--ffi-reg-sanity")) {
            ffi_reg_sanity = true;
        }
    }

    const mode = if (with_handler) "POSITIVE" else "NEGATIVE";
    try tty.print(.cyan, "=== M4.2 Custom Call Test ({s} MODE) ===\n", .{mode});

    {
        const tmp = std.process.getEnvVarOwned(allocator, "PJRT_GPU_PLUGIN_PATH") catch |e| switch (e) {
            error.EnvironmentVariableNotFound => null,
            inline else => return e,
        };
        defer if (tmp) |t| allocator.free(t);
        try tty.print(.yellow, "PJRT_GPU_PLUGIN_PATH: {?s}\n", .{tmp});
    }
    {
        const tmp = std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH") catch |e| switch (e) {
            error.EnvironmentVariableNotFound => null,
            inline else => return e,
        };
        defer if (tmp) |t| allocator.free(t);
        try tty.print(.yellow, "PJRT_PLUGIN_PATH: {?s}\n", .{tmp});
    }

    // Initialize MLIR context and dialects
    try stdout.print("Initializing MLIR context...\n", .{});

    var registry = try mlir.Registry.init();
    defer registry.deinit();

    mlir.DialectHandle.fromString("func").insertDialect(registry);
    mlir.DialectHandle.fromString("stablehlo").insertDialect(registry);

    var mlir_ctx = try mlir.Context.initWithRegistry(registry, false);
    defer mlir_ctx.deinit();

    mlir_ctx.allowUnregisteredDialects(false);

    const func_handle = mlir.DialectHandle.fromString("func");
    func_handle.registerDialect(mlir_ctx);
    _ = func_handle.loadDialect(mlir_ctx);

    const stablehlo_handle = mlir.DialectHandle.fromString("stablehlo");
    stablehlo_handle.registerDialect(mlir_ctx);
    _ = stablehlo_handle.loadDialect(mlir_ctx);

    if (!mlir_ctx.isRegisteredOperation("stablehlo.custom_call")) {
        return error.DialectRegistrationFailed;
    }

    try tty.print(.green, "Dialects registered\n", .{});

    // Build MLIR module with typed FFI custom call
    try stdout.print("Building MLIR module with typed FFI custom call...\n", .{});
    const bytecode = try buildModule(mlir_ctx, allocator);
    defer allocator.free(bytecode);

    try tty.print(.green, "Bytecode generated ({d} bytes)\n", .{bytecode.len});

    // Load PJRT plugin
    try stdout.print("Loading PJRT CUDA plugin...\n", .{});
    var plugin_path_owned: ?[]u8 = null;
    const plugin_path: []const u8 = blk: {
        if (std.process.getEnvVarOwned(allocator, "PJRT_GPU_PLUGIN_PATH")) |p| {
            plugin_path_owned = p;
            break :blk p;
        } else |_| {}
        break :blk "result/runtime/jax_plugins/xla_cuda13/xla_cuda_plugin.so";
    };
    try tty.print(.yellow, "Loading plugin: {s}\n", .{plugin_path});
    defer if (plugin_path_owned) |p| allocator.free(p);
    var api = try pjrt_plugin.loadPlugin(plugin_path);

    try tty.print(.green, "PJRT plugin loaded\n", .{});

    // Run registration sanity checks if requested
    if (ffi_reg_sanity) {
        try tty.print(.yellow, "Running FFI registration sanity checks...\n", .{});
        try testRegistrationSanity(api.pjrt_api);
        try tty.print(.green, "Sanity checks complete. Exiting.\n", .{});
        return;
    }

    // Register handler if in positive mode (before client creation).
    if (with_handler) {
        try stdout.print("Registering custom call handler...\n", .{});
        try registerHandler(api.pjrt_api, "cuda", size_variant);
        try tty.print(.green, "Handler registered via PJRT FFI extension\n", .{});
    } else {
        try stdout.print("Skipping handler registration (negative test)\n", .{});
    }

    // Create PJRT client (this initializes XLA service)
    try stdout.print("Creating PJRT client...\n", .{});
    var client_args = pjrt_api.initArgs(c.PJRT_Client_Create_Args);
    client_args.client = null;

    try api.call("PJRT_Client_Create", &client_args);
    const client = client_args.client orelse return error.ClientCreationFailed;
    defer {
        var destroy_args = pjrt_api.initArgs(c.PJRT_Client_Destroy_Args);
        destroy_args.client = client;
        api.call("PJRT_Client_Destroy", &destroy_args) catch {};
    }

    try tty.print(.green, "Client created\n", .{});

    // Query actual platform name from client
    var platform_args = pjrt_api.initArgs(c.PJRT_Client_PlatformName_Args);
    platform_args.client = client;
    try api.call("PJRT_Client_PlatformName", &platform_args);
    const actual_platform = platform_args.platform_name[0..platform_args.platform_name_size];
    try stdout.print("Actual platform name from client: '{s}'\n", .{actual_platform});

    // Compile program
    try stdout.print("Compiling program...\n", .{});

    // Create PJRT_Program struct
    var program = pjrt_api.initArgs(c.PJRT_Program);
    program.code = @constCast(bytecode.ptr);
    program.code_size = bytecode.len;
    program.format = "mlir".ptr;
    program.format_size = 4;

    // Minimal compile options (protobuf-encoded)
    const minimal_compile_opts = [_]u8{
        // CompileOptionsProto.executable_build_options (field 3, message)
        (3 << 3) | 2, 4, // tag 26, length 4

        // ExecutableBuildOptionsProto.num_replicas (field 4, int64) = 1
        (4 << 3) | 0, 0x01, // tag 32, value 1

        // ExecutableBuildOptionsProto.num_partitions (field 5, int64) = 1
        (5 << 3) | 0, 0x01, // tag 40, value 1
    };

    var compile_args = pjrt_api.initArgs(c.PJRT_Client_Compile_Args);
    compile_args.client = client;
    compile_args.program = &program;
    compile_args.compile_options = &minimal_compile_opts;
    compile_args.compile_options_size = minimal_compile_opts.len;
    compile_args.executable = null;

    api.call("PJRT_Client_Compile", &compile_args) catch |err| {
        if (!with_handler) {
            try tty.print(.yellow, "Compilation failed as expected (no handler): {any}\n", .{err});
            try tty.print(.green, "=== NEGATIVE TEST PASSED ===\n", .{});
            return;
        } else {
            try tty.print(.red, "Compilation failed unexpectedly: {any}\n", .{err});
            return err;
        }
    };

    const executable = compile_args.executable orelse return error.CompilationFailed;
    defer {
        var destroy_args = pjrt_api.initArgs(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.executable = executable;
        api.call("PJRT_LoadedExecutable_Destroy", &destroy_args) catch {};
    }

    try tty.print(.green, "Program compiled\n", .{});

    // Create input buffer with non-zero values
    try stdout.print("Creating input buffer...\n", .{});
    const input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // 2x3

    var buffer_args = pjrt_api.initArgs(c.PJRT_Client_BufferFromHostBuffer_Args);
    buffer_args.client = client;
    buffer_args.data = @ptrCast(@constCast(&input_data));
    buffer_args.type = c.PJRT_Buffer_Type_F32;
    buffer_args.dims = &[_]i64{ 2, 3 };
    buffer_args.num_dims = 2;
    buffer_args.buffer = null;

    try api.call("PJRT_Client_BufferFromHostBuffer", &buffer_args);
    const input_buffer = buffer_args.buffer orelse return error.BufferCreationFailed;
    defer {
        var destroy_args = pjrt_api.initArgs(c.PJRT_Buffer_Destroy_Args);
        destroy_args.buffer = input_buffer;
        api.call("PJRT_Buffer_Destroy", &destroy_args) catch {};
    }

    try tty.print(.green, "Input buffer created\n", .{});

    // Execute
    try stdout.print("Executing...\n", .{});

    // Allocate output buffer pointer (PJRT will fill this in)
    var output_ptrs = [_]?*c.PJRT_Buffer{null};
    const output_list: [*c]*c.PJRT_Buffer = @ptrCast(&output_ptrs);
    var output_lists = [_][*c]*c.PJRT_Buffer{output_list};

    var arg_list = [_]?*c.PJRT_Buffer{input_buffer};
    var arg_lists = [_][*c]?*c.PJRT_Buffer{&arg_list};

    var execute_args = pjrt_api.initArgs(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.executable = executable;
    execute_args.options = null;
    execute_args.num_devices = 1;
    execute_args.num_args = 1;
    execute_args.argument_lists = &arg_lists;
    execute_args.output_lists = @ptrCast(&output_lists);
    execute_args.device_complete_events = null;
    execute_args.execute_device = null;

    api.call("PJRT_LoadedExecutable_Execute", &execute_args) catch |err| {
        if (!with_handler) {
            try tty.print(.yellow, "Execution failed as expected (no handler): {any}\n", .{err});
            try tty.print(.green, "=== NEGATIVE TEST PASSED ===\n", .{});
            return;
        } else {
            try tty.print(.red, "Execution failed: {any}\n", .{err});
            return err;
        }
    };

    const output_buffer = output_ptrs[0] orelse return error.NullOutputBuffer;
    defer {
        var destroy_args = pjrt_api.initArgs(c.PJRT_Buffer_Destroy_Args);
        destroy_args.buffer = output_buffer;
        api.call("PJRT_Buffer_Destroy", &destroy_args) catch {};
    }

    try tty.print(.green, "Execution completed\n", .{});

    // Read output buffer
    try stdout.print("Reading output buffer...\n", .{});
    var output_data: [6]f32 = undefined;
    var tohost_args = pjrt_api.initArgs(c.PJRT_Buffer_ToHostBuffer_Args);
    tohost_args.src = output_buffer;
    tohost_args.dst = @ptrCast(&output_data);
    tohost_args.dst_size = @sizeOf(@TypeOf(output_data));

    try api.call("PJRT_Buffer_ToHostBuffer", &tohost_args);

    // Validate output is all zeros
    try stdout.print("Output values: [ ", .{});
    var all_zero = true;
    for (output_data) |val| {
        try stdout.print("{d:.1} ", .{val});
        if (val != 0.0) all_zero = false;
    }
    try stdout.print("]\n", .{});

    if (!all_zero) {
        try tty.print(.red, "ERROR: Expected all zeros, but got non-zero values\n", .{});
        return error.ValidationFailed;
    }

    try tty.print(.green, "Output validated: all zeros as expected\n", .{});
    try tty.print(.green, "=== POSITIVE TEST PASSED ===\n", .{});
}
