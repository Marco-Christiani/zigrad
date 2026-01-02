/// M1 Milestone Test: Basic PJRT Execution
const std = @import("std");
const zigrad = @import("zigrad");
const term_color = @import("util/term_color.zig");

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

    var tty = term_color.Tty.initForStderr(stdout);
    try stdout.print("Zigrad PJRT/XLA backend (M1)\n", .{});

    const plugin_path = std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH") catch |err| {
        try tty.print(.red, "error: PJRT_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        return err;
    };
    defer allocator.free(plugin_path);

    try stdout.print("Loading PJRT plugin from: {s}\n", .{plugin_path});

    var backend = PjrtBackend.init(allocator, plugin_path) catch |err| {
        try tty.print(.red, "Failed to initialize backend: {s}\n", .{@errorName(err)});
        return err;
    };
    defer backend.deinit();
    try tty.print(.green, "Backend initialized\n", .{});

    try stdout.print("Querying devices...\n", .{});
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
    try stdout.print("   Using device {d} ({s})\n", .{ device_id, @tagName(device_kind) });

    const program_text =
        \\func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;

    try stdout.print("Creating StableHLO program (elementwise add)...\n", .{});
    var program = try Program.fromBytecode(allocator, .mlir_text, program_text);
    defer program.deinit();
    try tty.print(.green, "Program created\n", .{});

    try stdout.print("Compiling program...\n", .{});
    const compile_options = zigrad.CompileOptions{
        .format = .stablehlo_mlir_text,
        .bytecode = program.bytecode,
        .optimization_level = 3,
        .dump_dir = null,
        .backend_options = null, // Let PJRT use defaults
    };

    var executable = backend.compile(device, compile_options) catch |err| {
        try tty.print(.red, "Compilation failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer executable.deinit();
    try tty.print(.green, "Compilation succeeded\n", .{});

    try stdout.print("Preparing input buffers...\n", .{});

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

    try stdout.writeAll("Uploading buffers to device...\n");
    const dev_buf_a = try backend.bufferFromHost(device, buf_a.data, .f32, shape);
    defer dev_buf_a.deinit();
    const dev_buf_b = try backend.bufferFromHost(device, buf_b.data, .f32, shape);
    defer dev_buf_b.deinit();

    const inputs = [_]zigrad.Buffer{ dev_buf_a, dev_buf_b };
    try tty.print(.green, "Buffers uploaded\n", .{});

    try stdout.writeAll("Executing program...\n");
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

    try stdout.writeAll("Reading results from device...\n");
    const output_buffer = result.outputs[0];

    var output_host = try HostBuffer.init(allocator, shape, .f32);
    defer output_host.deinit();

    var transfer_event = try output_buffer.toHost(output_host.data);
    defer transfer_event.deinit();

    try transfer_event.await_();

    try stdout.print("   Output:  ", .{});
    try output_host.print(stdout);
    try stdout.writeAll("\n");

    try stdout.writeAll("Verifying numerical correctness...\n");
    const output_slice = output_host.asSlice(f32);

    var all_correct = true;
    for (output_slice, 0..) |val, i| {
        const diff = @abs(val - expected[i]);
        if (diff > 1e-5) {
            try tty.print(.red, "Mismatch at index {d}: got {d:.2}, expected {d:.2}\n", .{ i, val, expected[i] });
            all_correct = false;
        }
    }

    if (all_correct) {
        try tty.print(.green, "All values correct\n", .{});
        try tty.print(.green, "M1 PASSED\n", .{});
    } else {
        try tty.print(.red, "M1 FAILED\n", .{});
        return error.NumericalMismatch;
    }
}
