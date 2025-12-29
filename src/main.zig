/// M1 Milestone Test: Basic PJRT Execution
///
/// Demonstrates:
/// 1. Loading PJRT plugin
/// 2. Compiling a simple StableHLO program (add op)
/// 3. Creating input buffers
/// 4. Executing the program
/// 5. Reading back results
/// 6. Verifying numerical correctness
const std = @import("std");
const zigrad = @import("zigrad");

const Backend = zigrad.Backend;
const HostBuffer = zigrad.HostBuffer;
const DType = zigrad.DType;
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

    try stdout.print("=== Zigrad PJRT/XLA Backend Prototype - M1 Test ===\n", .{});

    // Step 1: Get plugin path
    const plugin_path = std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH") catch |err| {
        try stdout.print("Error: PJRT_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        return err;
    };
    defer allocator.free(plugin_path);

    try stdout.print("1. Loading PJRT plugin from: {s}\n", .{plugin_path});

    // Step 2: Initialize backend
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

    // Step 4: Load or create program
    const program_text =
        \\func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;

    try stdout.print("3. Creating StableHLO program (elementwise add)...\n", .{});
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
        .backend_options = null, // Let PJRT use defaults
    };

    var executable = backend.compile(device, compile_options) catch |err| {
        try stdout.print("   ✗ Compilation failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer executable.deinit();
    try stdout.print("   ✓ Compilation succeeded\n", .{});

    // Step 6: Prepare inputs
    try stdout.print("5. Preparing input buffers...\n", .{});

    var input_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var input_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const expected = [_]f32{ 6.0, 8.0, 10.0, 12.0 };

    const shape = Shape{ .dims = &[_]usize{4} };

    var buf_a = try HostBuffer.fromSlice(allocator, &input_a, shape, .f32);
    defer buf_a.deinit();

    var buf_b = try HostBuffer.fromSlice(allocator, &input_b, shape, .f32);
    defer buf_b.deinit();

    try stdout.print("   Input A: ", .{});
    try buf_a.print(stdout);
    try stdout.print("   Input B: ", .{});
    try buf_b.print(stdout);
    try stdout.print("\n", .{});

    // Step 7: Upload to device
    try stdout.writeAll("6. Uploading buffers to device...\n");
    const dev_buf_a = try backend.bufferFromHost(device, buf_a.data, .f32, shape);
    defer dev_buf_a.deinit();
    const dev_buf_b = try backend.bufferFromHost(device, buf_b.data, .f32, shape);
    defer dev_buf_b.deinit();

    const inputs = [_]zigrad.Buffer{ dev_buf_a, dev_buf_b };
    try stdout.writeAll("   ✓ Buffers uploaded\n\n");

    // Step 8: Execute
    try stdout.writeAll("7. Executing program...\n");
    var result = executable.execute(&inputs, allocator) catch |err| {
        try stdout.print("   ✗ Execution failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer result.deinit(allocator);

    if (result.outputs.len == 0) {
        try stdout.writeAll("   ✗ No outputs returned\n");
        return error.NoOutputs;
    }

    try stdout.print("   ✓ Execution succeeded ({d} output(s))\n\n", .{result.outputs.len});

    // Step 9: Read back results
    try stdout.writeAll("8. Reading results from device...\n");
    const output_buffer = result.outputs[0];

    var output_host = try HostBuffer.init(allocator, shape, .f32);
    defer output_host.deinit();

    var transfer_event = try output_buffer.toHost(output_host.data);
    defer transfer_event.deinit();

    try transfer_event.await_();

    try stdout.print("   Output:  ", .{});
    try output_host.print(stdout);
    try stdout.writeAll("\n");

    // Step 10: Verify correctness
    try stdout.writeAll("9. Verifying numerical correctness...\n");
    const output_slice = output_host.asSlice(f32);

    var all_correct = true;
    for (output_slice, 0..) |val, i| {
        const diff = @abs(val - expected[i]);
        if (diff > 1e-5) {
            try stdout.print("   ✗ Mismatch at index {d}: got {d:.2}, expected {d:.2}\n", .{ i, val, expected[i] });
            all_correct = false;
        }
    }

    if (all_correct) {
        try stdout.writeAll("   ✓ All values correct!\n\n");
        try stdout.writeAll("=== M1 PASSED ===\n");
    } else {
        try stdout.writeAll("\n=== M1 FAILED ===\n");
        return error.NumericalMismatch;
    }
}
