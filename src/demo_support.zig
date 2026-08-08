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
    /// Write the PJRT executable's optimized HLO.
    dump_optimized_hlo: ?zg.pjrt.DumpOptimizedHlo = null,
    /// Write kernel-provider decisions after kernelization.
    dump_kernels: ?DumpKernels = null,
};

/// Selects kernel-provider diagnostics for scenarios that run kernelization.
pub const DumpKernels = struct {};

/// Applies the executable demos' default PJRT composition.
pub fn compile_pjrt(
    context: *PjrtContext,
    program: *zg.pr.Program,
    entry_name: []const u8,
    operations: PjrtOperations,
) !zg.Executor.LoadedProgram {
    var pipeline = try zg.pjrt.pipeline.create(
        context.compilation.allocator,
        &context.backend,
        .{
            .stablehlo = .{
                .entry_name = entry_name,
                .dump_pr = operations.dump_pr,
                .dump_stablehlo = operations.dump_stablehlo,
            },
            .dump_optimized_hlo = operations.dump_optimized_hlo,
        },
    );
    defer pipeline.deinit();

    return try pipeline.run(
        zg.Executor.LoadedProgram,
        program,
        &context.compilation,
    );
}
