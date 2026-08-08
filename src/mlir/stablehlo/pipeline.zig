//! Default validated PR to StableHLO pipeline segment.

const compilation = @import("../../compilation.zig");
const pr = @import("../../pr.zig");
const stablehlo = @import("../../stablehlo.zig");
const lower = @import("lower.zig");

/// Operations and lowering policy for the default StableHLO segment.
pub const Options = struct {
    /// PR function lowered as the StableHLO entry point.
    entry_name: []const u8,

    /// StableHLO serialization passed to the next operation.
    encoding: stablehlo.Encoding = .binary,
};

/// Append the default validated PR to StableHLO segment.
pub fn add(
    pipeline: *compilation.Pipeline,
    options: Options,
) compilation.Pipeline.AddError!void {
    try pipeline.add(pr.Validate{});
    try pipeline.add(lower.Lower{
        .config = .{
            .entry_name = options.entry_name,
            .encoding = options.encoding,
        },
    });
}
