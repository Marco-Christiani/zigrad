//! PJRT executable serialization round-trip witness.

const zg = @import("zigrad");
const demos = @import("demos.zig");

pub fn run(
    compilation_context: *zg.compilation.Context,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: *zg.pjrt.Backend,
) !void {
    const allocator = compilation_context.allocator;
    var program = try demos.build_demo_program(allocator);
    defer program.deinit();

    var pipeline = try zg.pjrt.pipeline.create(
        allocator,
        backend,
        .{ .stablehlo = .{ .entry_name = "main" } },
    );
    defer pipeline.deinit();
    var loaded_program = try pipeline.run(
        zg.Executor.LoadedProgram,
        &program,
        compilation_context,
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

    try demos.run_demo_executable(
        allocator,
        reloaded_program,
    );
}
