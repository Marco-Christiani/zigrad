//! IREE VM bytecode emission witness.

const std = @import("std");
const zg = @import("zigrad");
const cli = @import("../cli.zig");
const demos = @import("../demos.zig");

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    opts: cli.IreeCompileOpts,
    dump_mlir: ?*zg.output.Config,
) !void {
    var config = zg.iree.Config.from_environ(environ);
    if (opts.target) |target|
        config.compiler.target_backend = target;

    var program = try demos.build_demo_program(allocator);
    defer program.deinit();

    var compilation_context = zg.compilation.Context{
        .allocator = allocator,
        .io = io,
    };
    var compiler = zg.iree.Compiler{ .config = config.compiler };
    var pipeline = try zg.iree.pipeline.create(allocator, .{ .vmfb = &compiler }, .{
        .stablehlo = .{
            .entry_name = "main",
            .dump_stablehlo = if (dump_mlir) |selected| .{ .config = selected.* } else null,
        },
    });
    defer pipeline.deinit();
    var vmfb = try pipeline.run(
        zg.iree.Artifact,
        &program,
        &compilation_context,
    );
    defer vmfb.deinit();

    const output_path = opts.output orelse "demo.vmfb";
    try demos.write_bytes_to_path(io, output_path, vmfb.bytes);
    std.log.info("wrote {d} bytes VMFB -> {s}", .{ vmfb.bytes.len, output_path });
}
