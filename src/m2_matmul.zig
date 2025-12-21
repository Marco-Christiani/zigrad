/// M2 Milestone Test: Matmul with 2D Tensors
///
/// Tests:
/// - 2D tensor support (not just 1D vectors)
/// - Matrix multiplication via stablehlo.dot
/// - Numerical correctness of matrix operations
/// - Memory management for higher-rank tensors
const std = @import("std");
const zigrad = @import("zigrad");

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

    try stdout.print("=== Zigrad PJRT/XLA Backend Prototype - M2 Test ===\n", .{});

    // Step 1: Get plugin path from environment
    const plugin_path = std.process.getEnvVarOwned(allocator, "PJRT_CPU_PLUGIN_PATH") catch |err| {
        try stdout.print("Error: PJRT_CPU_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        try stdout.print("Please set PJRT_CPU_PLUGIN_PATH to the path of your PJRT CPU plugin.\n", .{});
        try stdout.print("Example: export PJRT_CPU_PLUGIN_PATH=/path/to/pjrt_cpu_plugin.so\n", .{});
        return err;
    };
    defer allocator.free(plugin_path);

    // Step 2: Initialize backend
    try stdout.print("1. Loading PJRT CPU plugin from: {s}\n", .{plugin_path});
    var backend = PjrtBackend.init(allocator, plugin_path) catch |err| {
        try stdout.print("   ✗ Failed to initialize backend: {s}\n", .{@errorName(err)});
        return err;
    };
    defer backend.deinit();
    try stdout.print("   ✓ Backend initialized\n", .{});

    // Step 3: Get devices
    try stdout.print("2. Querying devices...\n", .{});
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

    // Step 4: Create matrix multiplication program
    // A: 2x3, B: 3x2 -> C: 2x2
    const program_text =
        \\func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
        \\  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
        \\  return %0 : tensor<2x2xf32>
        \\}
    ;

    try stdout.print("3. Creating StableHLO program (matrix multiplication)...\n", .{});
    try stdout.print("   A: 2x3 matrix, B: 3x2 matrix -> C: 2x2 matrix\n", .{});
    var program = try Program.fromBytecode(allocator, .mlir_text, program_text);
    defer program.deinit();
    try stdout.print("   ✓ Program created\n", .{});

    // Step 5: Compile program
    try stdout.print("4. Compiling program...\n", .{});
    const compile_options = zigrad.CompileOptions{
        .format = .stablehlo_mlir_text,
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

    // Step 6: Prepare inputs
    try stdout.print("5. Preparing input matrices...\n", .{});

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
    // A @ B = [[1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12],
    //          [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]]
    //       = [[7 + 18 + 33,  8 + 20 + 36],
    //          [28 + 45 + 66, 32 + 50 + 72]]
    //       = [[58, 64],
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

    // Step 7: Upload to device
    try stdout.print("6. Uploading matrices to device...\n", .{});
    const dev_buf_a = try backend.bufferFromHost(device, buf_a.data, .f32, shape_a);
    defer dev_buf_a.deinit();
    const dev_buf_b = try backend.bufferFromHost(device, buf_b.data, .f32, shape_b);
    defer dev_buf_b.deinit();

    const inputs = [_]zigrad.Buffer{ dev_buf_a, dev_buf_b };
    try stdout.print("   ✓ Matrices uploaded\n\n", .{});

    // Step 8: Execute
    try stdout.print("7. Executing matrix multiplication...\n", .{});
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

    // Step 9: Read back results
    try stdout.print("8. Reading result matrix from device...\n", .{});
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

    // Step 10: Verify correctness
    try stdout.print("9. Verifying numerical correctness...\n", .{});
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
        try stdout.print("   ✓ All values correct!\n\n", .{});
        try stdout.print("=== M2 PASSED ===\n", .{});
    } else {
        try stdout.print("\n=== M2 FAILED ===\n", .{});
        return error.NumericalMismatch;
    }
}
