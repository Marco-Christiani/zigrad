/// Named symbol presets for terminal and text output.
pub const Symbols = struct {
    /// Marker used for successful outcomes.
    check: []const u8,

    /// Marker used for failed outcomes.
    x: []const u8,

    /// Multiplication symbol used in textual output.
    mul: []const u8,

    /// Right arrow used in textual output.
    right_arrow: []const u8,

    pub const unicode = Symbols{
        .check = "\u{2713}",
        .x = "\u{2717}",
        .mul = "\u{00d7}",
        .right_arrow = "\u{2192}",
    };

    pub const ascii = Symbols{
        .check = "[ok]",
        .x = "[x]",
        .mul = "x",
        .right_arrow = "->",
    };
};

test Symbols {
    const std = @import("std");

    try std.testing.expectEqualStrings("\u{2713}", Symbols.unicode.check);
    try std.testing.expectEqualStrings("x", Symbols.ascii.mul);
    try std.testing.expectEqualStrings("->", Symbols.ascii.right_arrow);
}
