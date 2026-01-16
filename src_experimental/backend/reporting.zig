const std = @import("std");
const backend = @import("backend.zig");
const pjrt_api = @import("../pjrt/api.zig");
const diagnostics = @import("../diagnostics.zig");

pub const CompileReport = struct {
    compile_key: []const u8,
    input_hash: []const u8,
    report_dir: []const u8,
    plugin_path: []const u8,
    xla_flags: ?[]const u8,
    cuda_data_dir: ?[]const u8,
    format: backend.CompileOptions.Format,
    optimization_level: u8,
    toolchain_major: usize,
    toolchain_minor: usize,
    unregistered_dialects: bool,

    pub fn deinit(self: *CompileReport, allocator: std.mem.Allocator) void {
        allocator.free(self.compile_key);
        allocator.free(self.input_hash);
        allocator.free(self.report_dir);
        allocator.free(self.plugin_path);
        if (self.xla_flags) |flags| allocator.free(flags);
        if (self.cuda_data_dir) |dir| allocator.free(dir);
    }
};

pub fn emitCompileReport(
    allocator: std.mem.Allocator,
    api: *pjrt_api.Api,
    plugin_path: []const u8,
    options: backend.CompileOptions,
) !CompileReport {
    const unregistered = diagnostics.isUnregisteredDialects();
    const toolchain_version = api.version();

    const input_hash = try sha256Hex(allocator, options.bytecode);
    const xla_flags = getEnvOwned(allocator, "XLA_FLAGS");
    const cuda_data_dir = try extractCudaDataDir(allocator, xla_flags);

    const compile_key = try computeCompileKey(
        allocator,
        options,
        xla_flags,
        cuda_data_dir,
    );

    const base_dir = options.dump_dir orelse "artifacts/compile";
    const report_dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, compile_key });
    try std.fs.cwd().makePath(report_dir);

    const plugin_path_owned = try allocator.dupe(u8, plugin_path);

    const report = CompileReport{
        .compile_key = compile_key,
        .input_hash = input_hash,
        .report_dir = report_dir,
        .plugin_path = plugin_path_owned,
        .xla_flags = xla_flags,
        .cuda_data_dir = cuda_data_dir,
        .format = options.format,
        .optimization_level = options.optimization_level,
        .toolchain_major = toolchain_version.major,
        .toolchain_minor = toolchain_version.minor,
        .unregistered_dialects = unregistered,
    };

    emitReportToStderr("compile", &report);
    try writeReportFile(report.report_dir, "compile_report.txt", &report);

    if (unregistered) {
        std.debug.print("warning: unverified/unregistered dialect mode is enabled\n", .{});
    }

    return report;
}

pub fn emitExecuteReport(report: *const CompileReport) void {
    emitReportToStderr("execute", report);
    writeReportFile(report.report_dir, "execute_report.txt", report) catch {};
    if (report.unregistered_dialects) {
        std.debug.print("warning: unverified/unregistered dialect mode is enabled\n", .{});
    }
}

fn emitReportToStderr(phase: []const u8, report: *const CompileReport) void {
    const format_tag = formatName(report.format);
    std.debug.print(
        "=== compile report ({s}) ===\n" ++
            "toolchain: PJRT {d}.{d}\n" ++
            "plugin_path: {s}\n" ++
            "format: {s}\n" ++
            "optimization_level: {d}\n" ++
            "input_hash: {s}\n" ++
            "compile_key: {s}\n",
        .{
            phase,
            report.toolchain_major,
            report.toolchain_minor,
            report.plugin_path,
            format_tag,
            report.optimization_level,
            report.input_hash,
            report.compile_key,
        },
    );

    if (report.xla_flags) |flags| {
        std.debug.print("XLA_FLAGS: {s}\n", .{flags});
    } else {
        std.debug.print("XLA_FLAGS: <unset>\n", .{});
    }

    if (report.cuda_data_dir) |dir| {
        std.debug.print("xla_gpu_cuda_data_dir: {s}\n", .{dir});
    } else {
        std.debug.print("xla_gpu_cuda_data_dir: <unset>\n", .{});
    }

    if (report.unregistered_dialects) {
        std.debug.print("mode: unverified/unregistered dialect\n", .{});
    }

    std.debug.print("report_dir: {s}\n", .{report.report_dir});
    std.debug.print("=== end compile report ===\n", .{});
}

fn writeReportFile(report_dir: []const u8, file_name: []const u8, report: *const CompileReport) !void {
    const path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}", .{ report_dir, file_name });
    defer std.heap.page_allocator.free(path);

    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    const format_tag = formatName(report.format);
    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    defer writer.flush() catch {};

    try writer.print("phase: {s}\n", .{file_name});
    try writer.print("toolchain: PJRT {d}.{d}\n", .{ report.toolchain_major, report.toolchain_minor });
    try writer.print("plugin_path: {s}\n", .{report.plugin_path});
    try writer.print("format: {s}\n", .{format_tag});
    try writer.print("optimization_level: {d}\n", .{report.optimization_level});
    try writer.print("input_hash: {s}\n", .{report.input_hash});
    try writer.print("compile_key: {s}\n", .{report.compile_key});
    if (report.xla_flags) |flags| {
        try writer.print("XLA_FLAGS: {s}\n", .{flags});
    } else {
        try writer.print("XLA_FLAGS: <unset>\n", .{});
    }
    if (report.cuda_data_dir) |dir| {
        try writer.print("xla_gpu_cuda_data_dir: {s}\n", .{dir});
    } else {
        try writer.print("xla_gpu_cuda_data_dir: <unset>\n", .{});
    }
    if (report.unregistered_dialects) {
        try writer.print("mode: unverified/unregistered dialect\n", .{});
    }
}

fn computeCompileKey(
    allocator: std.mem.Allocator,
    options: backend.CompileOptions,
    xla_flags: ?[]const u8,
    cuda_data_dir: ?[]const u8,
) ![]const u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(options.bytecode);
    hasher.update(formatName(options.format));
    hasher.update(&[_]u8{options.optimization_level});
    if (options.backend_options) |opts| {
        hasher.update(opts);
    }
    if (xla_flags) |flags| {
        hasher.update(flags);
    }
    if (cuda_data_dir) |dir| {
        hasher.update(dir);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return toHexOwned(allocator, &digest);
}

fn sha256Hex(allocator: std.mem.Allocator, data: []const u8) ![]const u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
    return toHexOwned(allocator, &digest);
}

fn toHexOwned(allocator: std.mem.Allocator, bytes: *const [32]u8) ![]const u8 {
    const hex_buf = std.fmt.bytesToHex(bytes.*, .lower);
    return allocator.dupe(u8, &hex_buf);
}

fn formatName(format: backend.CompileOptions.Format) []const u8 {
    return switch (format) {
        .stablehlo_portable => "stablehlo_portable",
        .stablehlo_mlir_text => "stablehlo_mlir_text",
        .stablehlo_mlir_bytecode => "stablehlo_mlir_bytecode",
        .mlir_text => "mlir_text",
        .mlir_bytecode => "mlir_bytecode",
    };
}

fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) ?[]const u8 {
    return std.process.getEnvVarOwned(allocator, name) catch null;
}

fn extractCudaDataDir(allocator: std.mem.Allocator, xla_flags: ?[]const u8) !?[]const u8 {
    if (std.process.getEnvVarOwned(allocator, "XLA_GPU_CUDA_DATA_DIR")) |dir| {
        return dir;
    } else |_| {}

    if (xla_flags) |flags| {
        const key = "--xla_gpu_cuda_data_dir=";
        if (std.mem.indexOf(u8, flags, key)) |idx| {
            const start = idx + key.len;
            const rest = flags[start..];
            const end = std.mem.indexOfAny(u8, rest, " \t\n") orelse rest.len;
            const dup = try allocator.dupe(u8, rest[0..end]);
            return dup;
        }
    }

    return null;
}
