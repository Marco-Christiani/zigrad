//! PJRT executable serialization round-trip witness.

const zg = @import("zigrad");
const demos = @import("demos.zig");
const demo_support = @import("demo_support.zig");

pub fn run(context: *demo_support.PjrtContext) !void {
    const allocator = context.compilation.allocator;
    const client = context.client;
    const execution = context.execution;
    var program = try demos.build_demo_program(allocator);
    defer program.deinit();

    var loaded_program = try demo_support.compile_pjrt(
        context,
        &program,
        "main",
        .{},
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
    var reloaded_program = try context.backend.loader.interface.load(&artifact);
    defer reloaded_program.deinit();

    try demos.run_demo_executable(
        allocator,
        reloaded_program,
    );
}
