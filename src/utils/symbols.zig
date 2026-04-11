//! Named symbol presets for terminal and text output.

/// Printable symbols
pub const Symbols = struct {
    /// Marker used for successful outcomes.
    check: []const u8,

    /// Marker used for failed outcomes.
    x: []const u8,

    /// Multiplication symbol.
    mul: []const u8,

    /// Arrow pointing right.
    right_arrow: []const u8,

    horiz: []const u8,
    vert_right: []const u8,
    up_right: []const u8,
    vert: []const u8,

    /// Whether tree connectors should include `horiz` after elbows/tees.
    ///
    /// For example, unicode trees typically use `├─` and `└─` while ASCII
    ///  output keeps `+` and `-` without an extra stroke, a consequence
    ///  of getting alignment right.
    tree_use_horiz: bool = false,

    pub const unicode = Symbols{
        .check = "\u{2713}",
        .x = "\u{2717}",
        .mul = "\u{00d7}",
        .right_arrow = "\u{2192}",
        // box drawings light horizontal
        .horiz = "\u{2500}",
        // box drawings light vertical and right
        .vert_right = "\u{251c}",
        // box drawings light up and right
        .up_right = "\u{2514}",
        // box drawing light vertical
        .vert = "\u{2502}",
        .tree_use_horiz = true,
    };

    pub const ascii = Symbols{
        .check = "[ok]",
        .x = "[x]",
        .mul = "x",
        .right_arrow = "->",
        .horiz = "-",
        .vert_right = "+",
        .up_right = "-",
        .vert = "|",
        .tree_use_horiz = false,
    };
};

test Symbols {
    const std = @import("std");

    try std.testing.expectEqualStrings("\u{2713}", Symbols.unicode.check);
    try std.testing.expectEqualStrings("x", Symbols.ascii.mul);
    try std.testing.expectEqualStrings("->", Symbols.ascii.right_arrow);
}
