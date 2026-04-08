//! Structured parameter tree with flat storage.
//!
//! Generic over leaf type. Stores leaves in DFS traversal order with
//!  parallel dot-separated paths. Structure is recoverable at comptime
//!  via `extract`.
//!
//! Useful for the spec -> host -> device buffer pipeline:
//!
//! ```zig
//! var specs = try Tree(Tensor).from(allocator, inputs_spec);
//! var host = try specs.map(Tensor, allocator, alloc_host);
//! var dev  = try host.map(Backend.Buffer, ctx, upload);
//! // dev.leaves[0..param_count] // ready for execution
//! ```
const std = @import("std");

pub fn Tree(comptime Leaf: type) type {
    return struct {
        const Self = @This();

        leaves: []Leaf,
        paths: []const []const u8,
        allocator: std.mem.Allocator,

        // ================================================================
        // Construction
        // ================================================================

        /// Build a Tree by flattening a typed struct value.
        ///
        /// Walks `value`'s fields at comptime, collecting leaves and their
        ///  dot-separated paths into parallel flat arrays. Paths are comptime
        ///  string literals (no allocation needed for them).
        pub fn from(allocator: std.mem.Allocator, value: anytype) !Self {
            const T = @TypeOf(value);
            const count = comptime leaf_count(T);
            const leaves = try allocator.alloc(Leaf, count);
            errdefer allocator.free(leaves);
            const paths = try allocator.alloc([]const u8, count);
            errdefer allocator.free(paths);

            // paths are comptime literals, just copy the pointers.
            const comptime_paths = comptime tree_paths(Leaf, T);
            @memcpy(paths, &comptime_paths);

            // Flatten values into the leaves array.
            var idx: usize = 0;
            flatten_values(T, value, leaves, &idx);

            return .{ .leaves = leaves, .paths = paths, .allocator = allocator };
        }

        /// Build from pre-existing parallel arrays (e.g., deserialization).
        /// Caller retains ownership of the underlying data, the tree borrows.
        pub fn from_slices(allocator: std.mem.Allocator, leaves: []Leaf, paths: []const []const u8) Self {
            std.debug.assert(leaves.len == paths.len);
            return .{ .leaves = leaves, .paths = paths, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.leaves);
            self.allocator.free(self.paths);
            self.* = undefined;
        }

        /// Deinit every leaf, then free the tree arrays.
        pub fn deinit_with(self: *Self, comptime deinit_fn: fn (*Leaf) void) void {
            for (self.leaves) |*leaf| deinit_fn(leaf);
            self.deinit();
        }

        // ================================================================
        // Structural recovery
        // ================================================================

        /// Reconstruct a typed value from the flat leaves.
        ///
        /// Inverse of `from`. Accepts any type that `from` can flatten:
        ///  named structs, tuples, arrays, or a bare `Leaf`. Paths are
        ///  not consulted - leaf order (DFS) determines field assignment.
        ///
        /// Handles types with comptime-inferred fields (e.g. anonymous tuples
        ///  built from module-level `const` values) by stripping `is_comptime`
        ///  so runtime leaf values can be assigned.
        pub fn extract(self: Self, comptime T: type) RuntimeOf(T) {
            const expected = comptime leaf_count(T);
            std.debug.assert(self.leaves.len == expected);
            var idx: usize = 0;
            return unflatten_values(T, self.leaves, &idx);
        }

        /// Flatten a typed value into a leaf array without paths or index.
        ///
        /// Useful when you only need the flat leaves (e.g. collecting output
        ///  tensors from a traced function). Caller owns the returned slice.
        pub fn flatten(allocator: std.mem.Allocator, value: anytype) ![]Leaf {
            const T = @TypeOf(value);
            const count = comptime leaf_count(T);
            const leaves = try allocator.alloc(Leaf, count);
            errdefer allocator.free(leaves);
            var idx: usize = 0;
            flatten_values(T, value, leaves, &idx);
            return leaves;
        }

        /// Reconstruct a typed value from a flat leaf slice (no tree needed).
        ///
        /// Standalone inverse of `flatten`. Useful when you have raw output
        ///  leaves (e.g. from execution) and want to recover structure without
        ///  constructing a full Tree.
        pub fn unflatten(comptime T: type, leaves: []const Leaf) RuntimeOf(T) {
            const expected = comptime leaf_count(T);
            std.debug.assert(leaves.len == expected);
            var idx: usize = 0;
            return unflatten_values(T, leaves, &idx);
        }

        // ================================================================
        // Transforms
        // ================================================================

        /// Apply a function to every leaf, producing a new tree with a
        ///  (possibly different) leaf type. Structure (paths) is preserved.
        ///
        /// The map function receives a context value and a leaf, returning
        ///  the transformed leaf. Use `{}` (void) for context-free transforms.
        pub fn map(
            self: Self,
            comptime NewLeaf: type,
            context: anytype,
            comptime mapFn: fn (@TypeOf(context), Leaf) anyerror!NewLeaf,
        ) !Tree(NewLeaf) {
            const new_leaves = try self.allocator.alloc(NewLeaf, self.leaves.len);
            errdefer self.allocator.free(new_leaves);
            for (self.leaves, 0..) |leaf, i| {
                new_leaves[i] = try mapFn(context, leaf);
            }
            const paths_copy = try self.allocator.dupe([]const u8, self.paths);
            return Tree(NewLeaf){ .leaves = new_leaves, .paths = paths_copy, .allocator = self.allocator };
        }

        /// Zip two trees with the same structure, producing a new tree.
        /// Both trees must have the same number of leaves.
        pub fn map2(
            self: Self,
            comptime OtherLeaf: type,
            other: *const Tree(OtherLeaf),
            comptime NewLeaf: type,
            context: anytype,
            comptime mapFn: fn (@TypeOf(context), Leaf, OtherLeaf) anyerror!NewLeaf,
        ) !Tree(NewLeaf) {
            std.debug.assert(self.leaves.len == other.leaves.len);
            const new_leaves = try self.allocator.alloc(NewLeaf, self.leaves.len);
            errdefer self.allocator.free(new_leaves);
            for (self.leaves, other.leaves, 0..) |a, b, i| {
                new_leaves[i] = try mapFn(context, a, b);
            }
            const paths_copy = try self.allocator.dupe([]const u8, self.paths);
            return Tree(NewLeaf){ .leaves = new_leaves, .paths = paths_copy, .allocator = self.allocator };
        }

        // ================================================================
        // Iteration
        // ================================================================

        /// Visit every leaf with its path. Visitor receives context,
        ///  path string, and a mutable pointer to the leaf.
        pub fn for_each(
            self: Self,
            context: anytype,
            comptime f: fn (@TypeOf(context), []const u8, *Leaf) void,
        ) void {
            for (self.paths, self.leaves) |path, *leaf| {
                f(context, path, leaf);
            }
        }

        /// Fold over all leaves.
        pub fn reduce(
            self: Self,
            comptime R: type,
            comptime f: fn (R, Leaf) R,
            init: R,
        ) R {
            var acc = init;
            for (self.leaves) |leaf| acc = f(acc, leaf);
            return acc;
        }

        // ================================================================
        // Runtime access
        // ================================================================

        /// Look up a leaf by dot-path. Linear scan.
        pub fn get(self: Self, path: []const u8) ?*Leaf {
            for (self.paths, 0..) |p, i| {
                if (std.mem.eql(u8, p, path)) return &self.leaves[i];
            }
            return null;
        }

        /// Look up a leaf by dot-path (const).
        pub fn get_const(self: Self, path: []const u8) ?Leaf {
            for (self.paths, 0..) |p, i| {
                if (std.mem.eql(u8, p, path)) return self.leaves[i];
            }
            return null;
        }

        /// Number of leaves.
        pub fn len(self: *const Self) usize {
            return self.leaves.len;
        }

        // ================================================================
        // Comptime helpers
        // ================================================================

        /// Count leaves in a struct type at comptime.
        /// TODO: this is kind of an unintuitive name, its not really operating
        ///  on what the user things the tree is.
        pub fn leaf_count(comptime T: type) comptime_int {
            if (T == Leaf) return 1;
            return switch (@typeInfo(T)) {
                .@"struct" => |info| blk: {
                    var total: comptime_int = 0;
                    for (info.fields) |field| total += leaf_count(field.type);
                    break :blk total;
                },
                .array => |info| info.len * leaf_count(info.child),
                else => @compileError("Tree(" ++ @typeName(Leaf) ++ "): unsupported type " ++ @typeName(T) ++ " - leaf types must be " ++ @typeName(Leaf) ++ ", structs, or arrays"),
            };
        }

        fn flatten_values(comptime T: type, value: T, out: []Leaf, idx: *usize) void {
            if (T == Leaf) {
                out[idx.*] = value;
                idx.* += 1;
                return;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    inline for (info.fields) |field| {
                        flatten_values(field.type, @field(value, field.name), out, idx);
                    }
                },
                .array => |info| {
                    inline for (0..info.len) |i| {
                        flatten_values(info.child, value[i], out, idx);
                    }
                },
                else => unreachable,
            }
        }

        fn unflatten_values(comptime T: type, leaves: []const Leaf, idx: *usize) RuntimeOf(T) {
            if (T == Leaf) {
                const val = leaves[idx.*];
                idx.* += 1;
                return val;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    var result: RuntimeOf(T) = undefined;
                    inline for (info.fields) |field| {
                        @field(result, field.name) = unflatten_values(field.type, leaves, idx);
                    }
                    return result;
                },
                .array => |info| {
                    var result: RuntimeOf(T) = undefined;
                    inline for (0..info.len) |i| {
                        result[i] = unflatten_values(info.child, leaves, idx);
                    }
                    return result;
                },
                else => unreachable,
            }
        }
    };
}

/// Strip `is_comptime` from struct fields so runtime values can be
///  stored. Returns `T` unchanged when no fields are comptime.
///
/// Module-level `const` structs passed into anonymous tuples get
///  comptime-typed fields. `extract` needs a runtime-assignable
///  version of the type to populate from the leaves array.
/// TODO: probably better to let it error or something because this
///  confuses zls a ton.
pub fn RuntimeOf(comptime T: type) type {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            for (info.fields) |field| {
                if (field.is_comptime) {
                    var fields: [info.fields.len]std.builtin.Type.StructField = undefined;
                    for (info.fields, 0..) |f, i| {
                        fields[i] = .{
                            .name = f.name,
                            .type = f.type,
                            .default_value_ptr = null,
                            .is_comptime = false,
                            .alignment = f.alignment,
                        };
                    }
                    return @Type(.{ .@"struct" = .{
                        .layout = info.layout,
                        .fields = &fields,
                        .decls = &.{},
                        .is_tuple = info.is_tuple,
                    } });
                }
            }
            return T;
        },
        else => return T,
    }
}

// ============================================================================
// Path generation - comptime dot-separated paths for any struct type.
// ============================================================================

/// Generate all dot-separated paths for a struct type with a given leaf type.
/// Returns a comptime array of string literals.
pub fn tree_paths(comptime Leaf: type, comptime T: type) [Tree(Leaf).leaf_count(T)][]const u8 {
    var result: [Tree(Leaf).leaf_count(T)][]const u8 = undefined;
    var idx: usize = 0;
    build_paths(Leaf, T, &result, &idx, "");
    return result;
}

fn build_paths(
    comptime Leaf: type,
    comptime T: type,
    result: [][]const u8,
    idx: *usize,
    comptime prefix: []const u8,
) void {
    if (T == Leaf) {
        result[idx.*] = prefix;
        idx.* += 1;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.fields) |field| {
                const sep = if (prefix.len == 0) "" else ".";
                build_paths(Leaf, field.type, result, idx, prefix ++ sep ++ field.name);
            }
        },
        .array => |info| {
            inline for (0..info.len) |i| {
                const sep = if (prefix.len == 0) "" else ".";
                build_paths(Leaf, info.child, result, idx, prefix ++ sep ++ std.fmt.comptimePrint("{d}", .{i}));
            }
        },
        else => @compileError("unsupported type in tree path: " ++ @typeName(T)),
    }
}

// ============================================================================
// Tests
// ============================================================================

test "from and extract round-trip" {
    const allocator = std.testing.allocator;

    const Spec = struct { dtype: u8, dim: i64 };
    const Params = struct {
        w: Spec,
        b: Spec,
    };
    const Batch = struct {
        x: Spec,
    };
    const Inputs = struct {
        params: Params,
        batch: Batch,
    };

    const inputs = Inputs{
        .params = .{ .w = .{ .dtype = 1, .dim = 784 }, .b = .{ .dtype = 1, .dim = 128 } },
        .batch = .{ .x = .{ .dtype = 1, .dim = 64 } },
    };

    // from works with any struct (including tuples)
    var tree = try Tree(Spec).from(allocator, inputs);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 3), tree.len());
    try std.testing.expectEqual(@as(i64, 784), tree.leaves[0].dim);
    try std.testing.expectEqual(@as(i64, 128), tree.leaves[1].dim);
    try std.testing.expectEqual(@as(i64, 64), tree.leaves[2].dim);

    // extract recovers named struct from flat leaves
    const recovered = tree.extract(Inputs);
    try std.testing.expectEqual(@as(i64, 784), recovered.params.w.dim);
    try std.testing.expectEqual(@as(i64, 128), recovered.params.b.dim);
    try std.testing.expectEqual(@as(i64, 64), recovered.batch.x.dim);
}

test "extract handles comptime-typed tuple fields" {
    const allocator = std.testing.allocator;

    const Inner = struct { a: i32, b: i32 };

    // Module-level const values produce comptime-typed anonymous tuple fields.
    const c1: Inner = .{ .a = 1, .b = 2 };
    const c2: Inner = .{ .a = 3, .b = 4 };
    const comptime_tuple = .{ c1, c2 };

    var tree = try Tree(i32).from(allocator, comptime_tuple);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 4), tree.len());

    // This would fail without RuntimeOf: "cannot store runtime value
    //  in compile time variable".
    const recovered = tree.extract(@TypeOf(comptime_tuple));
    try std.testing.expectEqual(@as(i32, 1), recovered.@"0".a);
    try std.testing.expectEqual(@as(i32, 2), recovered.@"0".b);
    try std.testing.expectEqual(@as(i32, 3), recovered.@"1".a);
    try std.testing.expectEqual(@as(i32, 4), recovered.@"1".b);
}

test "map transforms leaf type" {
    const allocator = std.testing.allocator;

    const Inner = struct { val: i32 };
    const S = struct { a: Inner, b: Inner };
    const input = S{ .a = .{ .val = 10 }, .b = .{ .val = 20 } };

    var tree = try Tree(Inner).from(allocator, input);
    defer tree.deinit();

    var mapped = try tree.map(i32, {}, struct {
        fn f(_: void, leaf: Inner) anyerror!i32 {
            return leaf.val * 2;
        }
    }.f);
    defer mapped.deinit();

    try std.testing.expectEqual(@as(usize, 2), mapped.len());
    try std.testing.expectEqual(@as(i32, 20), mapped.leaves[0]);
    try std.testing.expectEqual(@as(i32, 40), mapped.leaves[1]);
}

test "map with context" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };
    const input = S{ .a = 5, .b = 10 };

    var tree = try Tree(i32).from(allocator, input);
    defer tree.deinit();

    const scale: i32 = 3;
    var mapped = try tree.map(i32, scale, struct {
        fn f(s: i32, leaf: i32) anyerror!i32 {
            return leaf * s;
        }
    }.f);
    defer mapped.deinit();

    try std.testing.expectEqual(@as(i32, 15), mapped.leaves[0]);
    try std.testing.expectEqual(@as(i32, 30), mapped.leaves[1]);
}

test "map2" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };

    var t1 = try Tree(i32).from(allocator, S{ .a = 10, .b = 20 });
    defer t1.deinit();
    var t2 = try Tree(i32).from(allocator, S{ .a = 1, .b = 2 });
    defer t2.deinit();

    var result = try t1.map2(i32, &t2, i32, {}, struct {
        fn f(_: void, a: i32, b: i32) anyerror!i32 {
            return a + b;
        }
    }.f);
    defer result.deinit();

    try std.testing.expectEqual(@as(i32, 11), result.leaves[0]);
    try std.testing.expectEqual(@as(i32, 22), result.leaves[1]);
}

test "array fields" {
    const allocator = std.testing.allocator;

    const Layer = struct { w: i32, b: i32 };
    const Model = struct {
        embed: i32,
        layers: [3]Layer,
    };

    const model = Model{
        .embed = 100,
        .layers = .{
            .{ .w = 1, .b = 2 },
            .{ .w = 3, .b = 4 },
            .{ .w = 5, .b = 6 },
        },
    };

    var tree = try Tree(i32).from(allocator, model);
    defer tree.deinit();

    // embed + 3*(w+b) = 7
    try std.testing.expectEqual(@as(usize, 7), tree.len());
    try std.testing.expectEqual(@as(i32, 100), tree.leaves[0]);
    try std.testing.expectEqual(@as(i32, 1), tree.leaves[1]);
    try std.testing.expectEqual(@as(i32, 2), tree.leaves[2]);
    try std.testing.expectEqual(@as(i32, 5), tree.leaves[5]);
    try std.testing.expectEqual(@as(i32, 6), tree.leaves[6]);

    // Round-trip
    const recovered = tree.extract(Model);
    try std.testing.expectEqual(@as(i32, 100), recovered.embed);
    try std.testing.expectEqual(@as(i32, 3), recovered.layers[1].w);
    try std.testing.expectEqual(@as(i32, 6), recovered.layers[2].b);
}

test "get by path" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };
    var tree = try Tree(i32).from(allocator, S{ .a = 42, .b = 99 });
    defer tree.deinit();

    const a_ptr = tree.get("a");
    try std.testing.expect(a_ptr != null);
    try std.testing.expectEqual(@as(i32, 42), a_ptr.?.*);

    const b_ptr = tree.get("b");
    try std.testing.expect(b_ptr != null);
    try std.testing.expectEqual(@as(i32, 99), b_ptr.?.*);

    try std.testing.expect(tree.get("c") == null);
}

test "reduce" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32, c: i32 };
    var tree = try Tree(i32).from(allocator, S{ .a = 1, .b = 2, .c = 3 });
    defer tree.deinit();

    const sum = tree.reduce(i32, struct {
        fn f(acc: i32, leaf: i32) i32 {
            return acc + leaf;
        }
    }.f, 0);

    try std.testing.expectEqual(@as(i32, 6), sum);
}

test "for_each" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };
    var tree = try Tree(i32).from(allocator, S{ .a = 1, .b = 2 });
    defer tree.deinit();

    tree.for_each({}, struct {
        fn f(_: void, _: []const u8, leaf: *i32) void {
            leaf.* *= 10;
        }
    }.f);

    try std.testing.expectEqual(@as(i32, 10), tree.leaves[0]);
    try std.testing.expectEqual(@as(i32, 20), tree.leaves[1]);
}

test tree_paths {
    const Layer = struct { w: u8, b: u8 };
    const Model = struct {
        embed: u8,
        layers: [2]Layer,
        out: u8,
    };

    const paths = tree_paths(u8, Model);
    try std.testing.expectEqual(@as(usize, 6), paths.len);
    try std.testing.expectEqualStrings("embed", paths[0]);
    try std.testing.expectEqualStrings("layers.0.w", paths[1]);
    try std.testing.expectEqualStrings("layers.0.b", paths[2]);
    try std.testing.expectEqualStrings("layers.1.w", paths[3]);
    try std.testing.expectEqualStrings("layers.1.b", paths[4]);
    try std.testing.expectEqualStrings("out", paths[5]);
}

test "from populates paths correctly" {
    const allocator = std.testing.allocator;

    const Layer = struct { w: i32, b: i32 };
    const Model = struct {
        embed: i32,
        layers: [2]Layer,
    };

    var tree = try Tree(i32).from(allocator, Model{
        .embed = 0,
        .layers = .{ .{ .w = 0, .b = 0 }, .{ .w = 0, .b = 0 } },
    });
    defer tree.deinit();

    const expected = tree_paths(i32, Model);
    for (tree.paths, expected) |actual, exp| {
        try std.testing.expectEqualStrings(exp, actual);
    }
}

test "const tree supports leaf mutation and cleanup" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };

    // Leaf mutation works on a var binding -- only deinit requires var.
    {
        var tree = try Tree(i32).from(allocator, S{ .a = 1, .b = 2 });
        defer tree.deinit();

        // Mutate via get.
        tree.get("a").?.* = 100;
        try std.testing.expectEqual(@as(i32, 100), tree.leaves[0]);

        // Mutate via for_each.
        tree.for_each({}, struct {
            fn f(_: void, _: []const u8, leaf: *i32) void {
                leaf.* += 1;
            }
        }.f);
        try std.testing.expectEqual(@as(i32, 101), tree.leaves[0]);
        try std.testing.expectEqual(@as(i32, 3), tree.leaves[1]);

        // Direct leaf slice mutation.
        tree.leaves[1] = 999;
        try std.testing.expectEqual(@as(i32, 999), tree.get("b").?.*);
    }

    // deinit_with on const tree -- callback receives *Leaf for cleanup.
    {
        const Boxed = struct { val: []i32 };
        var tree = try Tree(Boxed).from(allocator, struct {
            a: Boxed,
            b: Boxed,
        }{
            .a = .{ .val = try allocator.dupe(i32, &.{ 1, 2 }) },
            .b = .{ .val = try allocator.dupe(i32, &.{ 3, 4 }) },
        });
        // deinit_with frees both the inner allocations and the tree arrays.
        tree.deinit_with(struct {
            fn f(leaf: *Boxed) void {
                std.testing.allocator.free(leaf.val);
            }
        }.f);
    }
}
