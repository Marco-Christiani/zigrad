//! Tree rendering utilities.
const std = @import("std");
const Symbols = @import("symbols.zig").Symbols;

pub const Options = struct {
    symbols: Symbols = .unicode,
};

fn write_tree_connector(writer: *std.Io.Writer, symbols: Symbols, is_last: bool) !void {
    const joint = if (is_last) symbols.up_right else symbols.vert_right;
    try writer.writeAll(joint);
    if (symbols.tree_use_horiz) {
        try writer.writeAll(symbols.horiz);
    }
}

fn write_depth_prefix(writer: *std.Io.Writer, symbols: Symbols, has_next: bool) !void {
    if (has_next) {
        if (symbols.tree_use_horiz) {
            try writer.print("{s}  ", .{symbols.vert});
        } else {
            try writer.print("{s} ", .{symbols.vert});
        }
        return;
    }

    if (symbols.tree_use_horiz) {
        try writer.writeAll("   ");
    } else {
        try writer.writeAll("  ");
    }
}

/// Render sorted paths as an indented tree diagram.
/// TODO: store in stable order rather than on demand?
pub fn render(
    allocator: std.mem.Allocator,
    paths: []const []const u8,
    writer: *std.Io.Writer,
    opts: Options,
) !void {
    if (paths.len == 0) {
        try writer.writeAll("(empty)\n");
        return;
    }

    const sorted = try allocator.dupe([]const u8, paths);
    defer allocator.free(sorted);

    std.sort.pdq([]const u8, sorted, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    var prev_parts: [128][]const u8 = undefined;
    var prev_len: usize = 0;

    for (sorted, 0..) |path, path_index| {
        var curr_parts: [128][]const u8 = undefined;
        const curr_len = split_path(path, &curr_parts);

        if (curr_len == 0) {
            try write_tree_connector(writer, opts.symbols, true);
            try writer.writeAll(" <root>\n");
            prev_len = 0;
            continue;
        }

        var common: usize = 0;
        while (common < prev_len and common < curr_len and
            std.mem.eql(u8, prev_parts[common], curr_parts[common]))
        {
            common += 1;
        }

        var i: usize = common;
        while (i < curr_len) : (i += 1) {
            var depth: usize = 0;
            while (depth < i) : (depth += 1) {
                const has_next = has_next_sibling(sorted, path_index, curr_parts[0..curr_len], depth);
                try write_depth_prefix(writer, opts.symbols, has_next);
            }
            const is_last = !has_next_sibling(sorted, path_index, curr_parts[0..curr_len], i);
            try write_tree_connector(writer, opts.symbols, is_last);
            try writer.print(" {s}\n", .{curr_parts[i]});
        }

        prev_len = curr_len;
        for (curr_parts[0..curr_len], 0..) |part, j| prev_parts[j] = part;
    }
}

fn split_path(path: []const u8, out: *[128][]const u8) usize {
    if (path.len == 0) return 0;
    var it = std.mem.splitScalar(u8, path, '.');
    var n: usize = 0;
    while (it.next()) |part| {
        if (n >= out.len) @panic("Tree path depth exceeds split buffer capacity");
        out[n] = part;
        n += 1;
    }
    return n;
}

fn has_next_sibling(
    sorted_paths: []const []const u8,
    current_index: usize,
    current_parts: []const []const u8,
    depth: usize,
) bool {
    var next_parts_buf: [128][]const u8 = undefined;
    for (sorted_paths[current_index + 1 ..]) |next_path| {
        const next_len = split_path(next_path, &next_parts_buf);
        if (next_len <= depth) continue;

        var parent_match = true;
        var i: usize = 0;
        while (i < depth) : (i += 1) {
            if (i >= current_parts.len or i >= next_len or
                !std.mem.eql(u8, current_parts[i], next_parts_buf[i]))
            {
                parent_match = false;
                break;
            }
        }
        if (!parent_match) continue;

        if (!std.mem.eql(u8, current_parts[depth], next_parts_buf[depth])) return true;
    }
    return false;
}

// ============================================================================
// Tests
// ============================================================================

test "render ascii tree" {
    const allocator = std.testing.allocator;
    const paths = [_][]const u8{ "a", "b.c", "b.d" };

    var out = std.io.Writer.Allocating.init(allocator);
    defer out.deinit();

    try render(allocator, &paths, &out.writer, .{ .symbols = Symbols.ascii });
    const results = try out.toOwnedSlice();
    defer allocator.free(results);
    try std.testing.expectEqualStrings(
        \\+ a
        \\- b
        \\  + c
        \\  - d
        \\
    ,
        results,
    );
}

test "render unicode tree" {
    const allocator = std.testing.allocator;
    const paths = [_][]const u8{ "a", "b.c", "b.d" };

    var out = std.io.Writer.Allocating.init(allocator);
    defer out.deinit();

    try render(allocator, &paths, &out.writer, .{ .symbols = Symbols.unicode });
    const results = try out.toOwnedSlice();
    defer allocator.free(results);
    try std.testing.expectEqualStrings(
        \\├─ a
        \\└─ b
        \\   ├─ c
        \\   └─ d
        \\
    ,
        results,
    );
}
