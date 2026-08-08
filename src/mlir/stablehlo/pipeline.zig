//! Default validated PR to StableHLO pipeline segment.

const compilation = @import("../../compilation.zig");
const pr = @import("../../pr.zig");
const stablehlo = @import("../../stablehlo.zig");
const lower = @import("lower.zig");

/// Operations and lowering policy for the default StableHLO segment.
pub const Options = struct {
    /// PR function lowered as the StableHLO entry point.
    entry_name: []const u8,

    /// Optional PR output pass applied after validation.
    dump_pr: ?pr.dump.Dump = null,

    /// Optional StableHLO output pass applied after lowering.
    dump_stablehlo: ?stablehlo.Dump = null,
};

/// Append the default validated PR to StableHLO segment.
pub fn add(
    pipeline: *compilation.Pipeline,
    options: Options,
) compilation.Pipeline.AddError!void {
    try pipeline.add(pr.Validate{});
    if (options.dump_pr) |selected| {
        var pass = selected;
        pass.config.entry_name = pass.config.entry_name orelse options.entry_name;
        try pipeline.add(pass);
    }
    try pipeline.add(lower.Lower{
        .config = .{
            .entry_name = options.entry_name,
            .encoding = if (options.dump_stablehlo == null) .binary else .text,
        },
    });
    if (options.dump_stablehlo) |selected| {
        var pass = selected;
        pass.config.entry_name = pass.config.entry_name orelse options.entry_name;
        try pipeline.add(pass);
    }
}
