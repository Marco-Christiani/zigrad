/// M2 Milestone: Get evidence of optimization (op fusion)
const std = @import("std");
const zigrad = @import("zigrad");
const term_color = @import("util/term_color.zig");

const HostBuffer = zigrad.HostBuffer;
const Shape = zigrad.Shape;
const Program = zigrad.Program;
const PjrtBackend = zigrad.pjrt_backend.PjrtBackend;

const c = @cImport({
    @cInclude("stdlib.h");
    @cInclude("unistd.h");
});

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const out = &stderr_writer.interface;
    defer out.flush() catch |e| switch (e) {
        error.WriteFailed => @panic("write failed on flush"),
    };

    var tty = term_color.Tty.initForStderr(out);
    try out.print("Zigrad PJRT/XLA backend (M2 fusion proof)\n", .{});

    const dump_cfg = try DumpConfig.init(allocator, out);
    defer dump_cfg.deinit(allocator);

    const plugin_path = try getPluginPath(allocator, out);
    defer allocator.free(plugin_path);

    try out.print("Loading PJRT plugin from: {s}\n", .{plugin_path});
    var backend = PjrtBackend.init(allocator, plugin_path) catch |err| {
        try tty.print(.red, "Failed to initialize backend: {s}\n", .{@errorName(err)});
        return err;
    };
    defer backend.deinit();
    try tty.print(.green, "Backend initialized\n", .{});

    try out.print("Querying devices...\n", .{});
    const devices = try backend.getDevices(allocator);
    defer {
        for (devices) |device| device.deinit();
        allocator.free(devices);
    }

    if (devices.len == 0) return error.NoDevices;
    const device = &devices[0];
    const device_kind = device.getKind();
    const device_id = try device.getId();
    try tty.print(.green, "Using device {d} ({s})\n", .{ device_id, @tagName(device_kind) });

    const shape = Shape{ .dims = &[_]usize{ 1024, 1024 } };
    const element_count: usize = 1024 * 1024;

    try out.print("Building StableHLO program (elementwise chain, f32[1024x1024])...\n", .{});
    const program_text = fusionProgramText();
    var program = try Program.fromBytecode(allocator, .mlir_text, program_text);
    defer program.deinit();
    try tty.print(.green, "Program created\n", .{});

    try out.print("Compiling program...\n", .{});
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

    if (dump_cfg.enabled) {
        try out.print("   Dump dir: {s}\n", .{dump_cfg.dump_dir});
    }

    try out.print("Preparing input tensors...\n", .{});
    const input_a = try allocator.alloc(f32, element_count);
    defer allocator.free(input_a);
    const input_b = try allocator.alloc(f32, element_count);
    defer allocator.free(input_b);
    const ref_out = try allocator.alloc(f32, element_count);
    defer allocator.free(ref_out);

    fillInputs(input_a, input_b);
    referenceCompute(ref_out, input_a, input_b);

    var host_a = try HostBuffer.fromSlice(allocator, input_a, shape, .f32);
    defer host_a.deinit();
    var host_b = try HostBuffer.fromSlice(allocator, input_b, shape, .f32);
    defer host_b.deinit();

    try out.print("Uploading tensors to device...\n", .{});
    const dev_a = try backend.bufferFromHost(device, host_a.data, .f32, shape);
    defer dev_a.deinit();
    const dev_b = try backend.bufferFromHost(device, host_b.data, .f32, shape);
    defer dev_b.deinit();
    const inputs = [_]zigrad.Buffer{ dev_a, dev_b };

    try out.print("Executing...\n", .{});
    var result = executable.execute(&inputs, allocator) catch |err| {
        try tty.print(.red, "Execution failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer result.deinit(allocator);
    if (result.outputs.len != 1) return error.UnexpectedOutputCount;

    try out.print("Reading output back...\n", .{});
    const output_buffer = result.outputs[0];
    const out_shape = output_buffer.getShape();
    if (out_shape.dims.len != 2 or out_shape.dims[0] != 1024 or out_shape.dims[1] != 1024) return error.ShapeMismatch;

    var host_out = try HostBuffer.init(allocator, shape, .f32);
    defer host_out.deinit();
    var transfer_event = try output_buffer.toHost(host_out.data);
    defer transfer_event.deinit();
    try transfer_event.await_();

    try out.print("Verifying correctness...\n", .{});
    const got = host_out.asSlice(f32);
    const stats = compareOutputs(got, ref_out);
    try out.print(
        "   max_abs_diff={d:.6} mean_abs_diff={d:.6} mismatches>tol({d:.4})={d}\n",
        .{ stats.max_abs_diff, stats.mean_abs_diff, stats.tolerance, stats.mismatch_count },
    );
    if (stats.mismatch_count != 0) return error.NumericalMismatch;

    if (dump_cfg.enabled) {
        try out.print("Dump summary...\n", .{});
        const summary = try summarizeXlaDumps(allocator, dump_cfg.dump_dir);
        try out.print(
            "   hlo_files_scanned={d} bytes_scanned={d} fusion_occurrences={d} elementwise_occurrences={d} fusion_per_elementwise={d:.4}\n",
            .{
                summary.hlo_files_scanned,
                summary.bytes_scanned,
                summary.fusion_occurrences,
                summary.elementwise_occurrences,
                summary.fusion_per_elementwise,
            },
        );
        if (summary.hlo_files_scanned == 0) {
            try out.print("   NOTE: no dump files found. Plugin may not honor XLA_FLAGS on CPU.\n", .{});
        }
    } else {
        try out.print("XLA dump disabled (set ZG_XLA_DUMP=1 to enable)\n", .{});
    }

    try tty.print(.green, "M2 FUSION PROOF PASS (correctness)\n", .{});
}

fn getPluginPath(allocator: std.mem.Allocator, out: anytype) ![]const u8 {
    if (std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH")) |p| {
        return p;
    } else |_| {}

    if (std.process.getEnvVarOwned(allocator, "PJRT_PLUGIN_PATH")) |p| {
        return p;
    } else |err| {
        try out.print("Error: PJRT_PLUGIN_PATH not set ({s})\n", .{@errorName(err)});
        return err;
    }
}

const DumpConfig = struct {
    enabled: bool,
    dump_dir: []const u8,
    xla_flags_value: ?[:0]u8,

    pub fn init(allocator: std.mem.Allocator, out: anytype) !DumpConfig {
        const enabled = envTruthy(allocator, "ZG_XLA_DUMP") catch false;
        if (!enabled) {
            return .{ .enabled = false, .dump_dir = &[_]u8{}, .xla_flags_value = null };
        }

        const dump_dir = try computeDumpDir(allocator);
        errdefer allocator.free(dump_dir);
        try std.fs.cwd().makePath(dump_dir);

        // Only set XLA_FLAGS when user explicitly opts in
        // we overwrite XLA_FLAGS because XLA only supports a single flags string, this ensures dumps land under dump_dir.
        const flags_noz = try std.fmt.allocPrint(
            allocator,
            "--xla_dump_to={s} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.* --xla_dump_hlo_module_re=.*",
            .{dump_dir},
        );
        defer allocator.free(flags_noz);

        const flags = try std.mem.concatWithSentinel(allocator, u8, &[_][]const u8{flags_noz}, 0);
        errdefer allocator.free(flags);

        if (c.setenv("XLA_FLAGS", flags.ptr, 1) != 0) {
            return error.SetEnvFailed;
        }

        try out.print("XLA dump enabled via XLA_FLAGS (ZG_XLA_DUMP=1)\n", .{});
        return .{ .enabled = true, .dump_dir = dump_dir, .xla_flags_value = flags };
    }

    pub fn deinit(self: DumpConfig, allocator: std.mem.Allocator) void {
        if (!self.enabled) return;
        allocator.free(self.dump_dir);
        if (self.xla_flags_value) |v| allocator.free(v);
    }
};

fn envTruthy(allocator: std.mem.Allocator, key: []const u8) !bool {
    const v = try std.process.getEnvVarOwned(allocator, key);
    defer allocator.free(v);
    if (v.len == 0) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.ascii.eqlIgnoreCase(v, "false")) return false;
    if (std.ascii.eqlIgnoreCase(v, "no")) return false;
    if (std.ascii.eqlIgnoreCase(v, "off")) return false;
    return true;
}

fn computeDumpDir(allocator: std.mem.Allocator) ![]const u8 {
    if (std.process.getEnvVarOwned(allocator, "ZG_XLA_DUMP_TO")) |override| {
        return override;
    } else |_| {}

    const ts: i64 = std.time.milliTimestamp();
    const pid: i32 = c.getpid();
    return std.fmt.allocPrint(allocator, "artifacts/xla_dumps/{d}-{d}", .{ ts, pid });
}

fn fusionProgramText() []const u8 {
    return 
    \\func.func @main(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    \\  %0 = stablehlo.add %a, %b : tensor<1024x1024xf32>
    \\  %1 = stablehlo.multiply %0, %b : tensor<1024x1024xf32>
    \\  %2 = stablehlo.subtract %1, %a : tensor<1024x1024xf32>
    \\  %3 = stablehlo.divide %2, %b : tensor<1024x1024xf32>
    \\  %4 = stablehlo.add %3, %0 : tensor<1024x1024xf32>
    \\  %5 = stablehlo.multiply %4, %4 : tensor<1024x1024xf32>
    \\  %6 = stablehlo.subtract %5, %1 : tensor<1024x1024xf32>
    \\  %7 = stablehlo.divide %6, %0 : tensor<1024x1024xf32>
    \\  %8 = stablehlo.add %7, %b : tensor<1024x1024xf32>
    \\  %9 = stablehlo.multiply %8, %2 : tensor<1024x1024xf32>
    \\  %10 = stablehlo.divide %9, %b : tensor<1024x1024xf32>
    \\  %11 = stablehlo.subtract %10, %7 : tensor<1024x1024xf32>
    \\  %12 = stablehlo.add %11, %5 : tensor<1024x1024xf32>
    \\  %13 = stablehlo.divide %12, %0 : tensor<1024x1024xf32>
    \\  %14 = stablehlo.multiply %13, %13 : tensor<1024x1024xf32>
    \\  %15 = stablehlo.add %14, %3 : tensor<1024x1024xf32>
    \\  %16 = stablehlo.subtract %15, %8 : tensor<1024x1024xf32>
    \\  %17 = stablehlo.multiply %16, %b : tensor<1024x1024xf32>
    \\  %18 = stablehlo.divide %17, %0 : tensor<1024x1024xf32>
    \\  %19 = stablehlo.add %18, %a : tensor<1024x1024xf32>
    \\  return %19 : tensor<1024x1024xf32>
    \\}
    ;
}

fn fillInputs(a: []f32, b: []f32) void {
    // Keep values positive and away from 0 to avoid division instabilities
    for (a, 0..) |*av, i| {
        const x = @as(f32, @floatFromInt(i % 1024)) / 1024.0;
        av.* = 0.5 + x; // [0.5, 1.5)
    }
    for (b, 0..) |*bv, i| {
        const x = @as(f32, @floatFromInt((i * 13) % 1024)) / 1024.0;
        bv.* = 1.0 + x; // [1.0, 2.0)
    }
}

fn referenceCompute(out: []f32, a: []const f32, b: []const f32) void {
    for (out, 0..) |*ov, i| {
        const aa = a[i];
        const bb = b[i];

        const t0 = aa + bb;
        const t1 = t0 * bb;
        const t2 = t1 - aa;
        const t3 = t2 / bb;
        const t4 = t3 + t0;
        const t5 = t4 * t4;
        const t6 = t5 - t1;
        const t7 = t6 / t0;
        const t8 = t7 + bb;
        const t9 = t8 * t2;
        const t10 = t9 / bb;
        const t11 = t10 - t7;
        const t12 = t11 + t5;
        const t13 = t12 / t0;
        const t14 = t13 * t13;
        const t15 = t14 + t3;
        const t16 = t15 - t8;
        const t17 = t16 * bb;
        const t18 = t17 / t0;
        const t19 = t18 + aa;

        ov.* = t19;
    }
}

const CompareStats = struct {
    max_abs_diff: f32,
    mean_abs_diff: f32,
    mismatch_count: usize,
    tolerance: f32,
};

fn compareOutputs(got: []const f32, expected: []const f32) CompareStats {
    const tol: f32 = 2e-3;
    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    var mismatches: usize = 0;

    for (got, 0..) |gv, i| {
        const ev = expected[i];
        const diff = @abs(gv - ev);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
        if (diff > tol) mismatches += 1;
    }

    return .{
        .max_abs_diff = max_abs,
        .mean_abs_diff = @floatCast(sum_abs / @as(f64, @floatFromInt(got.len))),
        .mismatch_count = mismatches,
        .tolerance = tol,
    };
}

const DumpSummary = struct {
    hlo_files_scanned: usize,
    bytes_scanned: u64,
    fusion_occurrences: u64,
    elementwise_occurrences: u64,
    fusion_per_elementwise: f64,
};

fn summarizeXlaDumps(allocator: std.mem.Allocator, dump_dir: []const u8) !DumpSummary {
    var dir = std.fs.cwd().openDir(dump_dir, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return .{
            .hlo_files_scanned = 0,
            .bytes_scanned = 0,
            .fusion_occurrences = 0,
            .elementwise_occurrences = 0,
            .fusion_per_elementwise = 0,
        },
        else => return err,
    };
    defer dir.close();

    var walker = try dir.walk(allocator);
    defer walker.deinit();

    var hlo_files_scanned: usize = 0;
    var bytes_scanned: u64 = 0;
    var fusion_occurrences: u64 = 0;
    var elementwise_occurrences: u64 = 0;

    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;

        // Only scan likely HLO text dumps:
        //  - filenames containing "hlo"
        //  - filenames ending in ".hlo"
        //  - filenames ending in ".txt"
        if (!isLikelyHloDump(entry.path)) continue;

        const file_bytes = readFileAlloc(dir, allocator, entry.path, 32 * 1024 * 1024) catch continue;
        defer allocator.free(file_bytes);

        // Require "HloModule" to avoid scanning unrelated dumps/logs.
        if (countSubstr(file_bytes, "HloModule") == 0) continue;

        hlo_files_scanned += 1;
        bytes_scanned += file_bytes.len;
        fusion_occurrences += countSubstr(file_bytes, "fusion");
        elementwise_occurrences += countElementwiseOps(file_bytes);
    }

    const ratio: f64 = if (elementwise_occurrences == 0)
        0
    else
        @as(f64, @floatFromInt(fusion_occurrences)) / @as(f64, @floatFromInt(elementwise_occurrences));

    return .{
        .hlo_files_scanned = hlo_files_scanned,
        .bytes_scanned = bytes_scanned,
        .fusion_occurrences = fusion_occurrences,
        .elementwise_occurrences = elementwise_occurrences,
        .fusion_per_elementwise = ratio,
    };
}

fn isLikelyHloDump(path: []const u8) bool {
    const base = std.fs.path.basename(path);
    if (std.mem.indexOf(u8, base, "hlo") != null) return true;
    if (std.mem.endsWith(u8, base, ".hlo")) return true;
    if (std.mem.endsWith(u8, base, ".txt")) return true;
    return false;
}

fn readFileAlloc(dir: std.fs.Dir, allocator: std.mem.Allocator, path: []const u8, max_bytes: usize) ![]u8 {
    var file = try dir.openFile(path, .{});
    defer file.close();
    return file.readToEndAlloc(allocator, max_bytes);
}

fn countElementwiseOps(hlo_text: []const u8) u64 {
    // HLO text usually renders elementwise ops like: `add(...)`, `multiply(...)`, etc
    // using a conservative substring match here to avoid counting unrelated tokens
    var total: u64 = 0;
    total += countSubstr(hlo_text, " add(");
    total += countSubstr(hlo_text, " multiply(");
    total += countSubstr(hlo_text, " subtract(");
    total += countSubstr(hlo_text, " divide(");
    total += countSubstr(hlo_text, " negate(");
    total += countSubstr(hlo_text, " exp(");
    total += countSubstr(hlo_text, " log(");
    return total;
}

fn countSubstr(haystack: []const u8, needle: []const u8) u64 {
    if (needle.len == 0 or haystack.len < needle.len) return 0;
    var i: usize = 0;
    var count: u64 = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) count += 1;
    }
    return count;
}
