//! Default PR to PJRT pipeline composition.

const std = @import("std");
const compilation = @import("../compilation.zig");
const stablehlo_pipeline = @import("../mlir/stablehlo/pipeline.zig");
const Backend = @import("backend.zig").Backend;
const DumpOptimizedHlo = @import("dump.zig").DumpOptimizedHlo;

/// Operations and lowering policy for the default PJRT pipeline.
pub const Options = struct {
    /// Validated PR to StableHLO segment.
    stablehlo: stablehlo_pipeline.Options,

    /// Optional optimized-HLO output pass applied after compilation and loading.
    dump_optimized_hlo: ?DumpOptimizedHlo = null,
};

/// Create the default validated PR to loaded PJRT program pipeline.
///
/// The returned queue owns copies of its operations. `backend_instance` and any
///  state referenced by optional operations must outlive the queue.
pub fn create(
    allocator: std.mem.Allocator,
    backend_instance: *Backend,
    options: Options,
) compilation.Pipeline.AddError!compilation.Pipeline {
    var result = compilation.Pipeline.init(allocator);
    errdefer result.deinit();

    try stablehlo_pipeline.append(&result, options.stablehlo);
    try result.add(&backend_instance.interface);
    if (options.dump_optimized_hlo) |selected| {
        var pass = selected;
        pass.config.entry_name = pass.config.entry_name orelse options.stablehlo.entry_name;
        try result.add(pass);
    }
    return result;
}
