//! IREE VM bytecode emission witness.

const std = @import("std");
const zg = @import("zigrad");
const demos = @import("../demos.zig");

pub const Options = struct {
    /// Destination for the emitted VMFB artifact.
    output: ?[]const u8 = null,

    /// IREE compilation target override.
    target: ?[]const u8 = null,

    /// Optional destination and format for PR output.
    pr: ?zg.pr.dump.Config = null,

    /// Optional destination for the MLIR passed to IREE.
    mlir: ?zg.output.Config = null,
};

pub fn run(
    io: std.Io,
    allocator: std.mem.Allocator,
    environ: *const std.process.Environ.Map,
    opts: Options,
) !void {
    var config = zg.iree.Config.from_environ(environ);
    if (opts.target) |target|
        config.compiler.target_backend = target;

    var program = try demos.build_demo_program(allocator);
    defer program.deinit();

    var ctx = zg.CompilationCtx{
        .allocator = allocator,
        .io = io,
    };
    var compiler = zg.iree.Compiler{ .config = config.compiler };
    var pipeline = zg.Pipeline.init(allocator);
    defer pipeline.deinit();
    try pipeline.add(zg.pr.Validate{});
    if (opts.pr) |selected| {
        var output_config = selected;
        output_config.entry_name = output_config.entry_name orelse "main";
        try pipeline.add(zg.pr.dump.Dump{ .config = output_config });
    }
    try pipeline.add(zg.pr.transform.outline.Pass{});
    try pipeline.add(zg.mlir.stablehlo.Lower{
        .config = .{
            .entry_name = "main",
            .encoding = if (opts.mlir == null) .binary else .text,
        },
    });
    if (opts.mlir) |selected| {
        var output_config = selected;
        output_config.entry_name = output_config.entry_name orelse "main";
        try pipeline.add(zg.stablehlo.Dump{ .config = output_config });
    }
    try pipeline.add(&compiler.interface);
    var vmfb = try pipeline.run(
        zg.iree.Artifact,
        &program,
        &ctx,
    );
    defer vmfb.deinit();

    const output_path = opts.output orelse "demo.vmfb";
    try demos.write_bytes_to_path(io, output_path, vmfb.bytes);
    std.log.info("wrote {d} bytes VMFB -> {s}", .{ vmfb.bytes.len, output_path });
}
