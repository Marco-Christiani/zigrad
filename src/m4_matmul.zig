/// M4 Milestone Test: In-Memory MLIR IR Construction
///
/// Tests:
/// - In-memory IR construction via MLIR C API (no string templates)
/// - Using StableHLO dialect via Zig APIs
/// - MLIR module serialization to bytecode
/// - Numerical correctness against M2
/// - Demonstrates full control over IR generation
///
/// This replaces M2's string-based IR with programmatic construction,
/// proving we can interpose compiler logic at the IR level.
const std = @import("std");
const zigrad = @import("zigrad");
const mlir = @import("mlir/mlir.zig");
const stablehlo = @import("mlir/dialects/stablehlo.zig");

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

    try stdout.print("=== Zigrad PJRT/XLA Backend Prototype - M4.1 Test ===\n", .{});
    try stdout.print("=== In-Memory MLIR IR Construction ===\n\n", .{});

    // Step 1: Get plugin path from environment
    const plugin_path = std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH") catch |err| {
        try stdout.print("Error: PJRT_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        try stdout.print("Please set PJRT_PLUGIN_PATH to the path of your PJRT plugin.\n", .{});
        return err;
    };
    defer allocator.free(plugin_path);

    // Step 2: Build MLIR IR in memory
    try stdout.print("1. Constructing MLIR IR in memory (via C API)...\n", .{});

    // Create MLIR context and allow unregistered dialects
    // This avoids needing to link the full StableHLO C++ library
    var mlir_ctx = try mlir.Context.init();
    defer mlir_ctx.deinit();
    mlir_ctx.allowUnregisteredDialects(true);

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

    try stdout.print("   ✓ IR constructed: func.func @main with stablehlo.dot\n", .{});

    // Print IR to verify (debug)
    if (false) {
        try stdout.print("\n--- Generated IR ---\n", .{});
        var print_buffer: [8192]u8 = undefined;
        var print_writer: std.Io.Writer = .fixed(&print_buffer);
        try module.op().print(&print_writer, .{});
        try stdout.print("{s}\n", .{print_writer.buffered()});
        try stdout.print("--- End IR ---\n\n", .{});
    }

    // Step 3: Serialize module to bytecode
    try stdout.print("2. Serializing MLIR module to bytecode...\n", .{});

    // serialize
    var bytecode_buffer_fixed: [1024 * 1024]u8 = undefined;
    var bytecode_writer: std.Io.Writer = .fixed(&bytecode_buffer_fixed);
    try module.op().writeBytecode(&bytecode_writer);

    const bytecode_buffer = try allocator.dupe(u8, bytecode_writer.buffered());
    defer allocator.free(bytecode_buffer);

    try stdout.print("   ✓ Bytecode generated ({d} bytes)\n", .{bytecode_buffer.len});

    // Step 4: Create Program from bytecode
    var program = try Program.fromBytecode(allocator, .mlir_bytecode, bytecode_buffer);
    defer program.deinit();

    // Step 5: Initialize backend
    try stdout.print("3. Loading PJRT plugin from: {s}\n", .{plugin_path});
    var backend = PjrtBackend.init(allocator, plugin_path) catch |err| {
        try stdout.print("   ✗ Failed to initialize backend: {s}\n", .{@errorName(err)});
        return err;
    };
    defer backend.deinit();
    try stdout.print("   ✓ Backend initialized\n", .{});

    // Step 6: Get devices
    try stdout.print("4. Querying devices...\n", .{});
    const devices = try backend.getDevices(allocator);
    defer {
        for (devices) |device| {
            device.deinit();
        }
        allocator.free(devices);
    }

    if (devices.len == 0) {
        try stdout.print("   ✗ No devices found\n", .{});
        return error.NoDevices;
    }

    try stdout.print("   ✓ Found {d} device(s)\n", .{devices.len});
    const device = &devices[0];
    const device_kind = device.getKind();
    const device_id = try device.getId();
    try stdout.print("   Using device {d} ({s})\n", .{ device_id, @tagName(device_kind) });

    // Step 7: Compile program
    try stdout.print("5. Compiling program...\n", .{});
    const compile_options = zigrad.CompileOptions{
        .format = .mlir_bytecode,
        .bytecode = program.bytecode,
        .optimization_level = 3,
        .dump_dir = null,
        .backend_options = null,
    };

    var executable = backend.compile(device, compile_options) catch |err| {
        try stdout.print("   ✗ Compilation failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer executable.deinit();
    try stdout.print("   ✓ Compilation succeeded\n", .{});

    // Step 8: Prepare inputs (same as M2)
    try stdout.print("6. Preparing input matrices...\n", .{});

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

    try stdout.print("   Matrix A (2x3): ", .{});
    try buf_a.print(stdout);
    try stdout.print("   Matrix B (3x2): ", .{});
    try buf_b.print(stdout);
    try stdout.print("\n", .{});

    // Step 9: Upload to device
    try stdout.print("7. Uploading matrices to device...\n", .{});
    const dev_buf_a = try backend.bufferFromHost(device, buf_a.data, .f32, shape_a);
    defer dev_buf_a.deinit();
    const dev_buf_b = try backend.bufferFromHost(device, buf_b.data, .f32, shape_b);
    defer dev_buf_b.deinit();

    const inputs = [_]zigrad.Buffer{ dev_buf_a, dev_buf_b };
    try stdout.print("   ✓ Matrices uploaded\n\n", .{});

    // Step 10: Execute
    try stdout.print("8. Executing matrix multiplication...\n", .{});
    var result = executable.execute(&inputs, allocator) catch |err| {
        try stdout.print("   ✗ Execution failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer result.deinit(allocator);

    if (result.outputs.len == 0) {
        try stdout.print("   ✗ No outputs returned\n", .{});
        return error.NoOutputs;
    }

    try stdout.print("   ✓ Execution succeeded ({d} output(s))\n\n", .{result.outputs.len});

    // Step 11: Read back results
    try stdout.print("9. Reading result matrix from device...\n", .{});
    const output_buffer = result.outputs[0];

    // Verify output shape
    const output_shape = output_buffer.getShape();
    if (output_shape.dims.len != 2 or output_shape.dims[0] != 2 or output_shape.dims[1] != 2) {
        try stdout.print("   ✗ Unexpected output shape: expected [2, 2], got [", .{});
        for (output_shape.dims, 0..) |dim, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{d}", .{dim});
        }
        try stdout.print("]\n", .{});
        return error.ShapeMismatch;
    }

    var output_host = try HostBuffer.init(allocator, shape_c, .f32);
    defer output_host.deinit();

    var transfer_event = try output_buffer.toHost(output_host.data);
    defer transfer_event.deinit();

    try transfer_event.await_();

    try stdout.print("   Result Matrix (2x2): ", .{});
    try output_host.print(stdout);
    try stdout.print("\n", .{});

    // Step 12: Verify correctness
    try stdout.print("10. Verifying numerical correctness vs M2 baseline...\n", .{});
    const output_slice = output_host.asSlice(f32);

    var all_correct = true;
    for (output_slice, 0..) |val, i| {
        const row = i / 2;
        const col = i % 2;
        const diff = @abs(val - expected[i]);
        if (diff > 1e-5) {
            try stdout.print("   ✗ Mismatch at [{d},{d}]: got {d:.2}, expected {d:.2}\n", .{ row, col, val, expected[i] });
            all_correct = false;
        }
    }

    if (all_correct) {
        try stdout.print("   ✓ All values match M2 baseline!\n\n", .{});
        try stdout.print("=== M4.1 PASSED ===\n", .{});
        try stdout.print("Success: In-memory IR construction produces identical results to M2\n", .{});
    } else {
        try stdout.print("\n=== M4.1 FAILED ===\n", .{});
        return error.NumericalMismatch;
    }
}
