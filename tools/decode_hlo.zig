const std = @import("std");
const xla = @import("xla_pb");
const hlo_decode = @import("hlo_decode");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("usage: decode_hlo <file.pb>\n", .{});
        std.process.exit(1);
    }

    const path = args[1];
    const bytes = blk: {
        var file = if (std.fs.path.isAbsolute(path))
            try std.fs.openFileAbsolute(path, .{})
        else
            try std.fs.cwd().openFile(path, .{});
        defer file.close();
        break :blk try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    };
    defer allocator.free(bytes);

    std.debug.print("read {d} bytes from {s}\n\n", .{ bytes.len, path });

    var buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&buffer);
    const out = &stdout_writer.interface;

    // Try HloModuleProtoWithConfig first (format from --dump-optimized),
    // fall back to plain HloModuleProto.
    blk: {
        var reader: std.Io.Reader = .fixed(bytes);
        const with_config = xla.HloModuleProtoWithConfig.decode(&reader, allocator) catch break :blk;
        defer @constCast(&with_config).deinit(allocator);

        if (with_config.hlo_module) |*module| {
            std.debug.print("(decoded as HloModuleProtoWithConfig)\n\n", .{});
            try hlo_decode.print_module(module, out);
            try out.flush();
            return;
        }
    }

    // Fall back to plain HloModuleProto
    var reader: std.Io.Reader = .fixed(bytes);
    const module = xla.HloModuleProto.decode(&reader, allocator) catch |err| {
        std.debug.print("decode error: {s}\n", .{@errorName(err)});
        std.process.exit(1);
    };
    defer @constCast(&module).deinit(allocator);

    std.debug.print("(decoded as HloModuleProto)\n\n", .{});
    try hlo_decode.print_module(&module, out);
    try out.flush();
}
