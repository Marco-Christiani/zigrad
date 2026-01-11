/// M2 Milestone Test: Matmul with 2D Tensors
const std = @import("std");
const zigrad = @import("zigrad");
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
    try tty.print(.white, "Zigrad PJRT/XLA backend (M2)\n", .{});

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

    // A: 2x3, B: 3x2 -> C: 2x2
    const program_text =
        \\func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
        \\  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
        \\  return %0 : tensor<2x2xf32>
        \\}
    ;

    try tty.print(.white, "Creating StableHLO program (matrix multiplication)...\n", .{});
    try tty.print(.white, "   A: 2x3 matrix, B: 3x2 matrix -> C: 2x2 matrix\n", .{});
    var program = try Program.fromBytecode(allocator, .mlir_text, program_text);
    defer program.deinit();
    try tty.print(.green, "Program created\n", .{});

    try tty.print(.white, "Compiling program...\n", .{});
    const compile_options = zigrad.CompileOptions{
        .format = .stablehlo_mlir_text,
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

    try tty.print(.white, "Verifying numerical correctness...\n", .{});
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
        try tty.print(.green, "All values correct\n", .{});
        try tty.print(.green, "M2 PASSED\n", .{});
    } else {
        try tty.print(.red, "M2 FAILED\n", .{});
        return error.NumericalMismatch;
    }
}
