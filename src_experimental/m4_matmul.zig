/// M4 Milestone Test: In-Memory MLIR IR Construction
const std = @import("std");
const zigrad = @import("zigrad");
const mlir = @import("mlir/mlir.zig");
const mlir_diag = @import("mlir/diagnostics.zig");
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

    var tty = term_color.Tty.initForStdout(stdout);
    try tty.print(.white, "Zigrad PJRT/XLA backend (M4.1)\n", .{});

    const plugin_path = std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH") catch |err| {
        try tty.print(.red, "error: PJRT_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        try tty.print(.red, "Set PJRT_PLUGIN_PATH to the path of your PJRT plugin.\n", .{});
        return err;
    };
    defer allocator.free(plugin_path);
    try tty.print(.white, "Loading PJRT plugin from: {s}\n", .{plugin_path});
    var backend = PjrtBackend.init(allocator, plugin_path) catch |err| {
        try tty.print(.red, "Failed to initialize backend: {s}\n", .{@errorName(err)});
        return err;
    };
    defer backend.deinit();
    try tty.print(.green, "Backend initialized\n", .{});

    try tty.print(.white, "Querying devices...\n", .{});
    const devices = try backend.getDevices(allocator);
    defer {
        for (devices) |device| {
            device.deinit();
        }
        allocator.free(devices);
    }

    if (devices.len == 0) {
        try tty.print(.red, "No devices found\n", .{});
        return error.NoDevices;
    }

    try tty.print(.green, "Found {d} device(s)\n", .{devices.len});
    const device = &devices[0];
    const device_kind = device.getKind();
    const device_id = try device.getId();
    try tty.print(.white, "   Using device {d} ({s})\n", .{ device_id, @tagName(device_kind) });

    try tty.print(.white, "Constructing MLIR IR in memory (via C API)...\n", .{});

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
    if (!mlir_ctx.isRegisteredOperation("stablehlo.dot")) return error.DialectRegistrationFailed;

    const diag = mlir_diag.checkStablehloDialectSupport();
    try tty.print(
        .red,
        "   ! Dialect registration diagnostics: issue={s} missing_libmlir_c={any} missing_libstablehlo_capi={any} symbol_in_process={any} symbol_in_libstablehlo_capi={any}\n",
        .{
            @tagName(diag.issue),
            diag.missing_libmlir_c,
            diag.missing_libstablehlo_capi,
            diag.symbol_in_process,
            diag.symbol_in_libstablehlo_capi,
        },
    );

    const loc = mlir.Location.unknown(mlir_ctx);

    // Create module
    var module = mlir.Module.init(loc);
    defer module.deinit();

    // Build function type: (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    const f32_type = mlir.Type.float(mlir_ctx, .f32);
    const arg0_type = mlir.Type.tensor(&.{ 2, 3 }, f32_type);
    const arg1_type = mlir.Type.tensor(&.{ 3, 2 }, f32_type);
    const result_type = mlir.Type.tensor(&.{ 2, 2 }, f32_type);

    const func_type = mlir.Type.function(mlir_ctx, &.{ arg0_type, arg1_type }, &.{result_type});

    // Create function body block first (with all operations inside)
    const entry_block = try mlir.Block.init(&.{ arg0_type, arg1_type }, &.{ loc, loc });

    // Get block arguments
    const arg0_val = entry_block.argument(0);
    const arg1_val = entry_block.argument(1);

    // Build stablehlo.dot operation
    const dot_op = mlir.Operation.make(mlir_ctx, "stablehlo.dot", .{
        .operands = &.{ arg0_val, arg1_val },
        .results = &.{result_type},
        .location = loc,
    });
    entry_block.appendOperation(dot_op);

    // Build return operation
    const return_op = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{dot_op.result(0)},
        // func.return verification expects it to be nested under func.func.
        // We construct the block before attaching it to the func.func op, so defer verification.
        .verify = false,
        .location = loc,
    });
    entry_block.appendOperation(return_op);

    // Create function operation with constructed block
    const func_op = mlir.Operation.make(mlir_ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, "main") },
            .{ "function_type", mlir.Attribute.type_(func_type) },
        },
        .location = loc,
    });

    // Add function to module
    module.getBody().appendOperation(func_op);

    // Final verification after assembly (we deferred verification of func.return until it is nested under func.func).
    if (!module.op().verify()) {
        return error.InvalidMlir;
    }

    try tty.print(.green, "IR constructed: func.func @main with stablehlo.dot\n", .{});

    // Print IR to verify (debug)
    if (false) {
        try tty.print(.white, "\n--- Generated IR ---\n", .{});
        var print_buffer: [8192]u8 = undefined;
        var print_writer: std.Io.Writer = .fixed(&print_buffer);
        try module.op().print(&print_writer, .{});
        try tty.print(.white, "{s}\n", .{print_writer.buffered()});
        try tty.print(.white, "--- End IR ---\n\n", .{});
    }

    try tty.print(.white, "Serializing MLIR module to bytecode...\n", .{});

    // serialize
    var bytecode_buffer_fixed: [1024 * 1024]u8 = undefined;
    var bytecode_writer: std.Io.Writer = .fixed(&bytecode_buffer_fixed);
    try module.op().writeBytecode(&bytecode_writer);

    const bytecode_buffer = try allocator.dupe(u8, bytecode_writer.buffered());
    defer allocator.free(bytecode_buffer);

    try tty.print(.green, "Bytecode generated ({d} bytes)\n", .{bytecode_buffer.len});

    var program = try Program.fromBytecode(allocator, .mlir_bytecode, bytecode_buffer);
    defer program.deinit();

    try tty.print(.white, "Compiling program...\n", .{});
    const compile_options = zigrad.CompileOptions{
        .format = .mlir_bytecode,
        .bytecode = program.bytecode,
        .optimization_level = 3,
        .dump_dir = null,
        .backend_options = null,
    };

    var executable = backend.compile(device, compile_options) catch |err| {
        try tty.print(.red, "Compilation failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer executable.deinit();
    try tty.print(.green, "Compilation succeeded\n", .{});

    try tty.print(.white, "Preparing input matrices...\n", .{});

    // Matrix A: 2x3
    // [[1, 2, 3],
    //  [4, 5, 6]]
    var input_a = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };

    // Matrix B: 3x2
    // [[7,  8],
    //  [9, 10],
    //  [11, 12]]
    var input_b = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
    };

    // Expected result: 2x2
    // A @ B = [[58, 64],
    //          [139, 154]]
    const expected = [_]f32{
        58.0,  64.0,
        139.0, 154.0,
    };

    const shape_a = Shape{ .dims = &[_]usize{ 2, 3 } };
    const shape_b = Shape{ .dims = &[_]usize{ 3, 2 } };
    const shape_c = Shape{ .dims = &[_]usize{ 2, 2 } };

    var buf_a = try HostBuffer.fromSlice(allocator, &input_a, shape_a, .f32);
    defer buf_a.deinit();

    var buf_b = try HostBuffer.fromSlice(allocator, &input_b, shape_b, .f32);
    defer buf_b.deinit();

    try tty.print(.white, "   Matrix A (2x3): ", .{});
    try buf_a.print(tty.writer);
    try tty.print(.white, "   Matrix B (3x2): ", .{});
    try buf_b.print(tty.writer);
    try tty.print(.white, "\n", .{});

    try tty.print(.white, "Uploading matrices to device...\n", .{});
    const dev_buf_a = try backend.bufferFromHost(device, buf_a.data, .f32, shape_a);
    defer dev_buf_a.deinit();
    const dev_buf_b = try backend.bufferFromHost(device, buf_b.data, .f32, shape_b);
    defer dev_buf_b.deinit();

    const inputs = [_]zigrad.Buffer{ dev_buf_a, dev_buf_b };
    try tty.print(.green, "Matrices uploaded\n", .{});

    try tty.print(.white, "Executing matrix multiplication...\n", .{});
    var result = executable.execute(&inputs, allocator) catch |err| {
        try tty.print(.red, "Execution failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer result.deinit(allocator);

    if (result.outputs.len == 0) {
        try tty.print(.red, "No outputs returned\n", .{});
        return error.NoOutputs;
    }

    try tty.print(.green, "Execution succeeded ({d} output(s))\n", .{result.outputs.len});

    try tty.print(.white, "Reading result matrix from device...\n", .{});
    const output_buffer = result.outputs[0];

    // Verify output shape
    const output_shape = output_buffer.getShape();
    if (output_shape.dims.len != 2 or output_shape.dims[0] != 2 or output_shape.dims[1] != 2) {
        try tty.print(.red, "Unexpected output shape: expected [2, 2], got [", .{});
        for (output_shape.dims, 0..) |dim, i| {
            if (i > 0) try tty.print(.white, ", ", .{});
            try tty.print(.white, "{d}", .{dim});
        }
        try tty.print(.white, "]\n", .{});
        return error.ShapeMismatch;
    }

    var output_host = try HostBuffer.init(allocator, shape_c, .f32);
    defer output_host.deinit();

    var transfer_event = try output_buffer.toHost(output_host.data);
    defer transfer_event.deinit();

    try transfer_event.await_();

    try tty.print(.white, "   Result Matrix (2x2): ", .{});
    try output_host.print(tty.writer);
    try tty.print(.white, "\n", .{});

    try tty.print(.white, "Verifying numerical correctness vs M2 baseline...\n", .{});
    const output_slice = output_host.asSlice(f32);

    var all_correct = true;
    for (output_slice, 0..) |val, i| {
        const row = i / 2;
        const col = i % 2;
        const diff = @abs(val - expected[i]);
        if (diff > 1e-5) {
            try tty.print(.red, "Mismatch at [{d},{d}]: got {d:.2}, expected {d:.2}\n", .{ row, col, val, expected[i] });
            all_correct = false;
        }
    }

    if (all_correct) {
        try tty.print(.green, "All values match M2 baseline\n", .{});
        try tty.print(.green, "M4.1 PASSED\n", .{});
    } else {
        try tty.print(.red, "M4.1 FAILED\n", .{});
        return error.NumericalMismatch;
    }
}
