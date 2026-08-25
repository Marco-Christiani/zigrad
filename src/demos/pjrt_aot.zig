//! PJRT executable serialization round-trip witness.

const zg = @import("zigrad");
const demos = @import("../demos.zig");

pub fn run(
    ctx: *zg.CompilationCtx,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
) !void {
    const allocator = ctx.allocator;
    var program = zg.pr.Program.init(allocator);
    defer program.deinit();
    const entry = try demos.build_demo_program(&program);
    try program.set_entry(entry);

    var pipeline = try zg.pjrt.pipeline.create(
        allocator,
        backend,
        .{},
    );
    defer pipeline.deinit();
    var loaded_program = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        ctx,
    );
    defer loaded_program.deinit();

    const serialized = try (try execution.loaded(loaded_program)).serialize(
        client.api,
        allocator,
    );
    defer allocator.free(serialized);

    var artifact = zg.pjrt.Artifact{
        .client = client,
        .loaded = try client.load_serialized_executable(serialized, null),
    };
    errdefer artifact.deinit();
    var reloaded_program = try backend.loader.interface.load(&artifact);
    defer reloaded_program.deinit();

    try demos.run_demo_executable(allocator, reloaded_program);
}
