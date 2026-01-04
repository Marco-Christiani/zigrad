/// M4.2 Milestone Test: Custom Call Boundaries
const std = @import("std");
const zigrad = @import("zigrad");
const mlir = @import("mlir/mlir.zig");
const stablehlo = @import("mlir/dialects/stablehlo.zig");
const term_color = @import("util/term_color.zig");

const Backend = zigrad.Backend;
const HostBuffer = zigrad.HostBuffer;
const Shape = zigrad.Shape;
const Program = zigrad.Program;
const PjrtBackend = zigrad.pjrt_backend.PjrtBackend;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stderr().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch |e| switch (e) {
        error.WriteFailed => @panic("write failed on flush"),
    };

    var tty = term_color.Tty.initForStderr(stdout);
    try stdout.print("Zigrad PJRT/XLA backend (M4.2 custom call)\n", .{});

    try stdout.print("Constructing MLIR IR with custom_call...\n", .{});

    var registry = try mlir.Registry.init();
    defer registry.deinit();

    // Dialect registry: register the dialects we will use in this test.
    // This should work without allowing unregistered dialects, provided the DSOs are linked and loadable:
    // - libMLIR-C.so
    // - libStablehloCAPI.so (exports mlirGetDialectHandle__stablehlo__)
    mlir.DialectHandle.fromString("func").insertDialect(registry);
    mlir.DialectHandle.fromString("stablehlo").insertDialect(registry);

    var mlir_ctx = try mlir.Context.initWithRegistry(registry, false);
    defer mlir_ctx.deinit();

    mlir_ctx.allowUnregisteredDialects(false);

    // Register + load dialects explicitly.
    const func_handle = mlir.DialectHandle.fromString("func");
    func_handle.registerDialect(mlir_ctx);
    _ = func_handle.loadDialect(mlir_ctx);

    const stablehlo_handle = mlir.DialectHandle.fromString("stablehlo");
    stablehlo_handle.registerDialect(mlir_ctx);
    _ = stablehlo_handle.loadDialect(mlir_ctx);

    // Correctness checks: ensure the context recognizes the ops we will create.
    if (!mlir_ctx.isRegisteredOperation("func.func")) return error.DialectRegistrationFailed;
    if (!mlir_ctx.isRegisteredOperation("func.return")) return error.DialectRegistrationFailed;
    if (!mlir_ctx.isRegisteredOperation("stablehlo.custom_call")) return error.DialectRegistrationFailed;

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

    // Build stablehlo.custom_call operation manually
    // This demonstrates an explicit boundary - XLA must call our custom implementation
    // Note: We construct this directly to avoid linking StablehloCAPI.a
    const custom_call_op = mlir.Operation.make(mlir_ctx, "stablehlo.custom_call", .{
        .operands = &.{arg0_val},
        .results = &.{output_type},
        .attributes = &.{
            .{ "call_target_name", mlir.Attribute.string(mlir_ctx, "zigrad_custom_relu") },
            .{ "has_side_effect", mlir.Attribute.boolean(mlir_ctx, false) },
            .{ "api_version", mlir.Attribute.int(mlir_ctx, .i32, 1) }, // original = 1
        },
        .location = loc,
    });
    entry_block.appendOperation(custom_call_op);

    // Build return operation
    const return_op = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{custom_call_op.result(0)},
        // func.return verification expects it to be nested under func.func.
        // We construct the block before attaching it to the func.func op, so defer verification.
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

    try tty.print(.green, "IR constructed with stablehlo.custom_call\n", .{});

    if (false) {
        try stdout.print("\n--- Generated IR ---\n", .{});
        var print_buffer: [8192]u8 = undefined;
        var print_writer: std.Io.Writer = .fixed(&print_buffer);
        try module.op().print(&print_writer, .{});
        try stdout.print("{s}\n", .{print_writer.buffered()});
        try stdout.print("--- End IR ---\n\n", .{});
    }

    try stdout.print("Serializing MLIR module to bytecode...\n", .{});

    var bytecode_buffer_fixed: [1024 * 1024]u8 = undefined;
    var bytecode_writer: std.Io.Writer = .fixed(&bytecode_buffer_fixed);
    try module.op().writeBytecode(&bytecode_writer);

    const bytecode_buffer = try allocator.dupe(u8, bytecode_writer.buffered());
    defer allocator.free(bytecode_buffer);

    try tty.print(.green, "Bytecode generated ({d} bytes)\n", .{bytecode_buffer.len});

    const has_custom_call = std.mem.indexOf(u8, bytecode_buffer, "zigrad_custom_relu") != null;
    if (has_custom_call) {
        try tty.print(.green, "Custom call target name found in bytecode\n", .{});
    } else {
        try tty.print(.red, "Custom call target name NOT found in bytecode\n", .{});
        return error.CustomCallMissing;
    }

    try tty.print(.green, "M4.2 PASSED\n", .{});
    try stdout.print("Note: Execution with PJRT requires registering a custom call handler.\n", .{});
}
