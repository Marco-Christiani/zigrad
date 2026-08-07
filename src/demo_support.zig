//! Shared runtime services and selected operations for executable demos.

const zg = @import("zigrad");

/// Services shared by executable scenarios that use PJRT.
pub const PjrtContext = struct {
    compilation: zg.compilation.Context,
    client: *zg.pjrt.Client,
    execution: *zg.pjrt.Execution,
    backend: zg.pjrt.Backend,
};

/// Optional operations composed by PJRT executable scenarios.
pub const PjrtOperations = struct {
    /// Write PR before lowering.
    dump_pr: ?zg.pr.dump.Dump = null,
    /// Write StableHLO after the selected StableHLO transforms.
    dump_stablehlo: ?zg.stablehlo.Dump = null,
    /// Write the PJRT executable's optimized representation.
    dump_optimized: ?zg.pjrt.DumpOptimized = null,
    /// Write kernel-provider decisions after kernelization.
    dump_kernels: ?DumpKernels = null,
};

/// Selects kernel-provider diagnostics for scenarios that run kernelization.
pub const DumpKernels = struct {};

/// Configuration for the demos' PR to StableHLO recipe.
pub const LowerStablehloOptions = struct {
    /// PR function to lower as the StableHLO entry point.
    entry_name: []const u8,
    /// Serialization encoding for the returned artifact.
    encoding: zg.stablehlo.Encoding = .binary,
    /// Optional PR output operation applied before lowering.
    dump_pr: ?zg.pr.dump.Dump = null,
};

/// Lower a demo PR program to StableHLO.
///
/// The caller releases the returned artifact with the context allocator.
pub fn lower_stablehlo(
    context: *zg.compilation.Context,
    program: *zg.pr.Program,
    options: LowerStablehloOptions,
) !zg.stablehlo.Artifact {
    var pr_flow = zg.compilation.start(program, context);
    try pr_flow.transform(zg.pr.Validate{});
    if (options.dump_pr) |selected| {
        var operation = selected;
        operation.config.entry_name = operation.config.entry_name orelse options.entry_name;
        try pr_flow.transform(operation);
    }

    const stablehlo_flow = try pr_flow.lower(zg.mlir.stablehlo.Lower{
        .config = .{
            .entry_name = options.entry_name,
            .encoding = options.encoding,
        },
    });
    return stablehlo_flow.value;
}

/// Applies the executable demos' default PJRT composition.
pub fn compile_pjrt(
    context: *PjrtContext,
    program: *zg.pr.Program,
    entry_name: []const u8,
    operations: PjrtOperations,
) !zg.Executor.LoadedProgram {
    var stablehlo_flow = zg.compilation.start(
        try lower_stablehlo(&context.compilation, program, .{
            .entry_name = entry_name,
            .encoding = if (operations.dump_stablehlo == null) .binary else .text,
            .dump_pr = operations.dump_pr,
        }),
        &context.compilation,
    );
    defer stablehlo_flow.value.deinit(context.compilation.allocator);

    if (operations.dump_stablehlo) |selected| {
        var operation = selected;
        operation.config.entry_name = operation.config.entry_name orelse entry_name;
        try stablehlo_flow.transform(operation);
    }

    var loaded_program_flow = try stablehlo_flow.compile(&context.backend.interface);
    errdefer loaded_program_flow.value.deinit();
    if (operations.dump_optimized) |selected| {
        var operation = selected;
        operation.config.entry_name = operation.config.entry_name orelse entry_name;
        try loaded_program_flow.transform(operation);
    }
    return loaded_program_flow.value;
}
