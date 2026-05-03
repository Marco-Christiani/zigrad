//! Structured parameter tree with flat storage and fast path lookups.
//!
//! `Tree(Leaf)` stores leaves in DFS order with parallel dot-separated paths.
//! Paths are dot-separated (e.g. `layers.10.mlp.down_proj`).
//!
//! ## Ownership model
//!
//! - `Tree` owns its leaf array, path strings, and path index.
//! - `from_slices` copies inputs - caller retains ownership of the originals.
//! - `subtree`/`subtree_glob` return non-owning views backed by a parent tree.
//! - Views mutate parent leaves in place and only own view metadata.
//!
//! ## Usage
//!
//! ```zig
//! var specs = try Tree(Tensor).from(allocator, inputs_spec);
//! var host = try specs.map(Tensor, allocator, alloc_host);
//! var dev  = try host.map(Tensor, ctx, upload);
//! const params = try dev.extract(Params);
//! ```
const std = @import("std");
const meta = @import("meta.zig");
const tree_render = @import("tree_render.zig");

pub const RuntimeOf = meta.RuntimeOf;
pub const TreeRenderOptions = tree_render.Options;

pub fn Tree(comptime Leaf: type) type {
    return struct {
        const Self = @This();
        const PathIndex = std.StringHashMapUnmanaged(usize);

        // TODO: Replace `parent: *Self` with `root_leaves: []Leaf` and
        //  `root_paths: []const []const u8` for move-safety. Nested views
        //  should compose indices eagerly at creation.
        const View = struct {
            parent: *Self,
            indices: []usize,
            base_prefix: ?[]const u8,
            index: PathIndex,
        };

        leaves: []Leaf,
        paths: []const []const u8,
        allocator: std.mem.Allocator,
        index: PathIndex,
        view: ?View = null,

        // ================================================================
        // Construction / destruction
        // ================================================================

        /// Build a tree by flattening a typed struct value.
        ///
        /// Walks `value`'s fields at comptime, collecting leaves and their
        ///  dot-separated paths. The tree owns all allocations.
        pub fn from(allocator: std.mem.Allocator, value: anytype) !Self {
            const T = @TypeOf(value);
            // NOTE: this could be free, since we know size at comptime but for a large model this
            //  would mean a non-trivial amount of stack space not acceptable on edge targets. A
            //  comptime path may make sense, but seems like a premature over-optimization right
            //  now. The same comments apply to all the functions that allow temp buffers.
            const count = comptime leaf_count(T);

            const leaves = try allocator.alloc(Leaf, count);
            errdefer allocator.free(leaves);

            var idx: usize = 0;
            meta.flatten(Leaf, T, value, leaves, &idx);

            const comptime_paths = comptime meta.tree_paths(Leaf, T);
            const owned_paths = try clone_paths(allocator, &comptime_paths);
            errdefer free_paths(allocator, owned_paths);

            return make_owned(Leaf, allocator, leaves, owned_paths);
        }

        /// Build from pre-existing parallel slices. Data is copied.
        ///
        /// Caller retains ownership of the originals.
        /// Returns `error.DuplicatePath` when `paths` contains duplicates.
        pub fn from_slices(
            allocator: std.mem.Allocator,
            leaves: []const Leaf,
            paths: []const []const u8,
        ) !Self {
            std.debug.assert(leaves.len == paths.len);

            const leaves_copy = try allocator.dupe(Leaf, leaves);
            errdefer allocator.free(leaves_copy);

            const paths_copy = try clone_paths(allocator, paths);
            errdefer free_paths(allocator, paths_copy);

            return make_owned(Leaf, allocator, leaves_copy, paths_copy);
        }

        pub fn deinit(self: *Self) void {
            if (self.view) |*v| {
                if (v.base_prefix) |p| self.allocator.free(p);
                v.index.deinit(self.allocator);
                self.allocator.free(v.indices);
            } else {
                self.index.deinit(self.allocator);
                self.allocator.free(self.leaves);
                free_paths(self.allocator, self.paths);
            }
            self.* = undefined;
        }

        /// Deinit every leaf, then free tree storage.
        ///
        /// Valid only for owned trees. Views do not own leaf storage.
        /// TODO: based on usage pattern thus far, we can probably drop this
        ///  as a separate method and just call the type's declared deinit
        ///  method, at least by default.
        pub fn deinit_with(self: *Self, comptime deinit_fn: fn (*Leaf) void) void {
            std.debug.assert(self.view == null);
            for (self.leaves) |*leaf| deinit_fn(leaf);
            self.deinit();
        }

        // ================================================================
        // Comptime helpers (delegate to meta)
        // ================================================================

        /// Count leaves of type `Leaf` in struct type `T` at comptime.
        /// TODO: Static method - no tree instance needed, doesnt belong here.
        pub fn leaf_count(comptime T: type) comptime_int {
            return meta.leaf_count(Leaf, T);
        }

        // ================================================================
        // Structural recovery
        // ================================================================

        /// Reconstruct a typed value from visible leaves.
        ///
        /// Inverse of `from`. Reconstruction is positional: leaf order (DFS)
        ///  determines field assignment and path names are not consulted.
        /// When called on a view, this allocates a temporary contiguous leaf
        ///  buffer before unflattening.
        /// Handles comptime-typed fields via `RuntimeOf`.
        pub fn extract(self: *const Self, comptime T: type) !RuntimeOf(T) {
            const expected = comptime leaf_count(T);
            std.debug.assert(self.len() == expected);

            if (self.view == null) {
                var idx: usize = 0;
                return meta.unflatten(Leaf, T, self.leaves, &idx);
            }

            // View: gather visible leaves into contiguous buffer.
            const tmp = try self.allocator.alloc(Leaf, self.len());
            defer self.allocator.free(tmp);
            for (0..self.len()) |i| {
                tmp[i] = self.visible_leaf_value(i);
            }
            var idx: usize = 0;
            return meta.unflatten(Leaf, T, tmp, &idx);
        }

        /// Flatten a typed value into a leaf array without paths.
        ///
        /// Caller owns the slice.
        /// TODO: Static method - no tree instance needed, doesnt belong here.
        pub fn flatten(allocator: std.mem.Allocator, value: anytype) ![]Leaf {
            const T = @TypeOf(value);
            const count = comptime leaf_count(T);
            const leaves = try allocator.alloc(Leaf, count);
            errdefer allocator.free(leaves);
            var idx: usize = 0;
            meta.flatten(Leaf, T, value, leaves, &idx);
            return leaves;
        }

        /// Reconstruct a typed value from a flat leaf slice (no tree needed).
        ///
        /// Static inverse of `flatten`.
        pub fn unflatten(comptime T: type, leaves: []const Leaf) RuntimeOf(T) {
            const expected = comptime leaf_count(T);
            std.debug.assert(leaves.len == expected);
            var idx: usize = 0;
            return meta.unflatten(Leaf, T, leaves, &idx);
        }

        // ================================================================
        // Transforms
        // ================================================================

        /// Map every leaf to a new type, preserving paths.
        ///
        /// `map_fn` receives a context value and a leaf, returning the
        ///  transformed leaf. Use `{}` (void) for context-free transforms.
        /// TODO: removed the rather clever comptime types here, but it allowed
        ///  users to not have to make dummy types just to use these functions and
        ///  frankly I miss that. Goes for all fns accepting callbacks.
        pub fn map(
            self: *const Self,
            comptime NewLeaf: type,
            context: anytype,
            comptime map_fn: fn (@TypeOf(context), Leaf) anyerror!NewLeaf,
        ) !Tree(NewLeaf) {
            const n = self.len();
            const new_leaves = try self.allocator.alloc(NewLeaf, n);
            errdefer self.allocator.free(new_leaves);

            for (0..n) |i| {
                new_leaves[i] = try map_fn(context, self.visible_leaf_value(i));
            }

            const paths_copy = try self.clone_visible_paths();
            errdefer free_paths(self.allocator, paths_copy);

            return make_owned(NewLeaf, self.allocator, new_leaves, paths_copy);
        }

        /// Zip two trees and map leaf pairs.
        ///
        /// **Currently, pairs are matched by visible position after a
        ///  length check, not by path equality.**
        ///
        /// Returns `error.TreeLengthMismatch` when visible lengths differ.
        pub fn map2(
            self: *const Self,
            comptime OtherLeaf: type,
            other: *const Tree(OtherLeaf),
            comptime NewLeaf: type,
            context: anytype,
            comptime map_fn: fn (@TypeOf(context), Leaf, OtherLeaf) anyerror!NewLeaf,
        ) !Tree(NewLeaf) {
            if (self.len() != other.len()) return error.TreeLengthMismatch;

            const new_leaves = try self.allocator.alloc(NewLeaf, self.len());
            errdefer self.allocator.free(new_leaves);

            for (0..self.len()) |i| {
                new_leaves[i] = try map_fn(
                    context,
                    self.visible_leaf_value(i),
                    other.visible_leaf_value(i),
                );
            }

            const paths_copy = try self.clone_visible_paths();
            errdefer free_paths(self.allocator, paths_copy);

            return make_owned(NewLeaf, self.allocator, new_leaves, paths_copy);
        }

        // ================================================================
        // Iteration
        // ================================================================

        /// Visit every leaf with its path.
        pub fn for_each(
            self: *Self,
            context: anytype,
            comptime f: fn (@TypeOf(context), []const u8, *Leaf) void,
        ) void {
            for (0..self.len()) |i| {
                f(context, self.visible_path(i), self.visible_leaf_ptr(i));
            }
        }

        /// Visit leaves under `prefix` only.
        ///
        /// Prefix semantics are path-segment aware: matches `prefix` itself
        ///  or any path starting with `prefix.`.
        pub fn for_each_prefix(
            self: *Self,
            prefix: []const u8,
            context: anytype,
            comptime f: fn (@TypeOf(context), []const u8, *Leaf) void,
        ) void {
            for (0..self.len()) |i| {
                const path = self.visible_path(i);
                if (!path_matches_prefix(path, prefix)) continue;
                f(context, path, self.visible_leaf_ptr(i));
            }
        }

        /// Visit leaves matching a segment-based dot-path glob.
        ///
        /// Glob syntax: `*` matches one segment, `**` matches zero or more.
        /// Wildcards must be full segments (not partial text).
        pub fn for_each_glob(
            self: *Self,
            glob: []const u8,
            context: anytype,
            comptime f: fn (@TypeOf(context), []const u8, *Leaf) void,
        ) !void {
            const parsed = try ParsedGlob.init(self.allocator, glob);
            defer parsed.deinit(self.allocator);

            for (0..self.len()) |i| {
                const path = self.visible_path(i);
                if (!path_matches_glob_parts(path, parsed.parts)) continue;
                f(context, path, self.visible_leaf_ptr(i));
            }
        }

        /// Fold over leaves in DFS order.
        pub fn reduce(
            self: *const Self,
            comptime R: type,
            comptime f: fn (R, Leaf) R,
            init: R,
        ) R {
            var acc = init;
            for (0..self.len()) |i| {
                acc = f(acc, self.visible_leaf_value(i));
            }
            return acc;
        }

        // ================================================================
        // Runtime path operations
        // ================================================================

        /// Number of visible leaves.
        pub fn len(self: *const Self) usize {
            return if (self.view) |v| v.indices.len else self.leaves.len;
        }

        /// Check if a path exists in the current visible namespace.
        pub fn contains(self: *const Self, path: []const u8) bool {
            return self.lookup_visible_index(path) != null;
        }

        /// Get mutable leaf ptr, lookup by visible path.
        ///
        /// Returns `error.PathNotFound` when `path` is missing.
        pub fn getptr(self: *Self, path: []const u8) !*Leaf {
            const idx = self.lookup_visible_index(path) orelse return error.PathNotFound;
            return self.visible_leaf_ptr(idx);
        }

        /// Get leaf by value, lookup by visible path.
        ///
        /// Returns `error.PathNotFound` when `path` is missing.
        pub fn get(self: *const Self, path: []const u8) !Leaf {
            const idx = self.lookup_visible_index(path) orelse return error.PathNotFound;
            return self.visible_leaf_value(idx);
        }

        /// Create a non-owning subtree view under `prefix`.
        ///
        /// The returned view borrows this tree's storage. The parent tree must
        ///  outlive the view. Mutations through the view update parent leaves
        ///  in place.
        ///
        /// Prefix semantics are segment-aware: matches `prefix` itself and
        ///  descendants under `prefix.`.
        ///
        /// Returns `error.PrefixNotFound` when no visible path matches.
        pub fn subtree(self: *Self, prefix: []const u8) !Self {
            var count: usize = 0;
            for (0..self.len()) |i| {
                if (path_matches_prefix(self.visible_path(i), prefix)) count += 1;
            }
            if (count == 0) return error.PrefixNotFound;

            const indices = try self.allocator.alloc(usize, count);
            errdefer self.allocator.free(indices);

            var filled: usize = 0;
            for (0..self.len()) |i| {
                if (!path_matches_prefix(self.visible_path(i), prefix)) continue;
                indices[filled] = i;
                filled += 1;
            }

            const base_prefix = try self.allocator.dupe(u8, prefix);
            errdefer self.allocator.free(base_prefix);

            return self.make_view(indices, base_prefix);
        }

        /// Create a non-owning subtree view matching a segment-based glob.
        ///
        /// The returned view borrows this tree's storage. The parent tree must
        ///  outlive the view. Mutations through the view update parent leaves
        ///  in place.
        ///
        /// Glob syntax: `*` matches one segment, `**` matches zero or more.
        /// Wildcards must be whole segments.
        ///
        /// Returns `error.PatternNotFound` when no visible path matches.
        pub fn subtree_glob(self: *Self, glob: []const u8) !Self {
            const parsed = try ParsedGlob.init(self.allocator, glob);
            defer parsed.deinit(self.allocator);

            var count: usize = 0;
            for (0..self.len()) |i| {
                if (path_matches_glob_parts(self.visible_path(i), parsed.parts)) count += 1;
            }
            if (count == 0) return error.PatternNotFound;

            const indices = try self.allocator.alloc(usize, count);
            errdefer self.allocator.free(indices);

            var filled: usize = 0;
            for (0..self.len()) |i| {
                if (!path_matches_glob_parts(self.visible_path(i), parsed.parts)) continue;
                indices[filled] = i;
                filled += 1;
            }

            return self.make_view(indices, null);
        }

        // ================================================================
        // Printing
        // ================================================================

        /// Write a tree diagram of visible paths.
        ///
        /// Allocates a temporary path slice for rendering.
        pub fn render(self: *const Self, writer: *std.Io.Writer, opts: TreeRenderOptions) !void {
            const display_paths = try self.allocator.alloc([]const u8, self.len());
            defer self.allocator.free(display_paths);

            for (0..self.len(), display_paths) |i, *dp| dp.* = self.visible_path(i);

            try tree_render.render(self.allocator, display_paths, writer, opts);
        }

        pub fn print(self: *const Self, writer: *std.Io.Writer) !void {
            for (0..self.len()) |i| {
                try writer.print("{s}: {any}\n", .{ self.visible_path(i), self.visible_leaf_value(i) });
            }
        }

        // ================================================================
        // Internal helpers
        // ================================================================

        fn make_owned(
            comptime L: type,
            allocator: std.mem.Allocator,
            leaves: []L,
            paths: []const []const u8,
        ) !Tree(L) {
            // caller owns leaves/paths
            var idx_map: std.StringHashMapUnmanaged(usize) = .empty;
            errdefer idx_map.deinit(allocator);

            for (paths, 0..) |path, i| {
                const gop = try idx_map.getOrPut(allocator, path);
                if (gop.found_existing) return error.DuplicatePath;
                gop.value_ptr.* = i;
            }

            return .{
                .leaves = leaves,
                .paths = paths,
                .allocator = allocator,
                .index = idx_map,
                .view = null,
            };
        }

        fn make_view(self: *Self, indices: []usize, base_prefix: ?[]const u8) !Self {
            errdefer self.allocator.free(indices);
            errdefer if (base_prefix) |p| self.allocator.free(p);

            var sel_index: PathIndex = .empty;
            errdefer sel_index.deinit(self.allocator);

            for (indices, 0..) |local_idx, i| {
                const p = self.visible_path(local_idx);
                const shown = if (base_prefix) |prefix|
                    relative_path_checked(p, prefix) orelse return error.PathOutsidePrefix
                else
                    p;
                const gop = try sel_index.getOrPut(self.allocator, shown);
                if (gop.found_existing) return error.DuplicatePath;
                gop.value_ptr.* = i;
            }

            return .{
                .leaves = &.{},
                .paths = &.{},
                .allocator = self.allocator,
                .index = .empty,
                .view = .{
                    .parent = self,
                    .indices = indices,
                    .base_prefix = base_prefix,
                    .index = sel_index,
                },
            };
        }

        fn lookup_visible_index(self: *const Self, path: []const u8) ?usize {
            if (self.view) |v| return v.index.get(path);
            return self.index.get(path);
        }

        fn visible_path(self: *const Self, idx: usize) []const u8 {
            if (self.view) |v| {
                const parent_path = v.parent.visible_path(v.indices[idx]);
                return if (v.base_prefix) |prefix|
                    relative_path_checked(parent_path, prefix) orelse parent_path
                else
                    parent_path;
            }
            return self.paths[idx];
        }

        fn visible_leaf_ptr(self: *Self, idx: usize) *Leaf {
            if (self.view) |v| {
                return v.parent.visible_leaf_ptr(v.indices[idx]);
            }
            return &self.leaves[idx];
        }

        fn visible_leaf_value(self: *const Self, idx: usize) Leaf {
            if (self.view) |v| {
                return v.parent.visible_leaf_value(v.indices[idx]);
            }
            return self.leaves[idx];
        }

        fn clone_visible_paths(self: *const Self) ![]const []const u8 {
            const out = try self.allocator.alloc([]const u8, self.len());
            var filled: usize = 0;
            errdefer {
                for (out[0..filled]) |p| self.allocator.free(p);
                self.allocator.free(out);
            }
            for (0..self.len()) |i| {
                out[i] = try self.allocator.dupe(u8, self.visible_path(i));
                filled = i + 1;
            }
            return out;
        }

        fn clone_paths(allocator: std.mem.Allocator, src: []const []const u8) ![]const []const u8 {
            const out = try allocator.alloc([]const u8, src.len);
            var filled: usize = 0;
            errdefer {
                for (out[0..filled]) |p| allocator.free(p);
                allocator.free(out);
            }
            for (src, 0..) |path, i| {
                out[i] = try allocator.dupe(u8, path);
                filled = i + 1;
            }
            return out;
        }

        fn free_paths(allocator: std.mem.Allocator, paths: []const []const u8) void {
            for (paths) |path| allocator.free(path);
            allocator.free(paths);
        }

        fn path_matches_prefix(path: []const u8, prefix: []const u8) bool {
            if (prefix.len == 0) return true;
            if (std.mem.eql(u8, path, prefix)) return true;
            if (path.len <= prefix.len) return false;
            return std.mem.startsWith(u8, path, prefix) and path[prefix.len] == '.';
        }

        fn relative_path_checked(path: []const u8, prefix: []const u8) ?[]const u8 {
            if (prefix.len == 0) return path;
            if (std.mem.eql(u8, path, prefix)) return "";
            if (!path_matches_prefix(path, prefix)) return null;
            return path[prefix.len + 1 ..];
        }

        /// This is NOT a very strong implementation of glob. Quick and dirty
        ///  to serve its purpose here, but limited.
        const ParsedGlob = struct {
            parts: []const []const u8,

            fn init(allocator: std.mem.Allocator, glob: []const u8) !ParsedGlob {
                var parts_buf: [128][]const u8 = undefined;
                const part_count = split_path(glob, &parts_buf);
                if (part_count == 0) return error.InvalidGlobPattern;

                for (parts_buf[0..part_count]) |seg| {
                    if (seg.len == 0) return error.InvalidGlobPattern;
                    if (std.mem.indexOfScalar(u8, seg, '*') != null and
                        !std.mem.eql(u8, seg, "*") and
                        !std.mem.eql(u8, seg, "**"))
                    {
                        return error.InvalidGlobPattern;
                    }
                }

                const owned_parts = try allocator.alloc([]const u8, part_count);
                for (parts_buf[0..part_count], 0..) |p, i| owned_parts[i] = p;
                return .{ .parts = owned_parts };
            }

            fn deinit(self: ParsedGlob, allocator: std.mem.Allocator) void {
                allocator.free(self.parts);
            }
        };

        /// Split a dot-path into segments using a fixed temporary buffer.
        ///
        /// Panics if segment count exceeds `out.len`.
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

        fn path_matches_glob_parts(path: []const u8, glob_parts: []const []const u8) bool {
            var path_parts_buf: [128][]const u8 = undefined;
            const path_len = split_path(path, &path_parts_buf);
            return match_glob_parts(glob_parts, path_parts_buf[0..path_len], 0, 0);
        }

        fn match_glob_parts(
            glob_parts: []const []const u8,
            path_parts: []const []const u8,
            glob_i: usize,
            path_i: usize,
        ) bool {
            if (glob_i == glob_parts.len) return path_i == path_parts.len;

            const seg = glob_parts[glob_i];
            if (std.mem.eql(u8, seg, "**")) {
                var next_glob_i = glob_i + 1;
                while (next_glob_i < glob_parts.len and std.mem.eql(u8, glob_parts[next_glob_i], "**")) {
                    next_glob_i += 1;
                }
                if (next_glob_i == glob_parts.len) return true;

                var i = path_i;
                while (i <= path_parts.len) : (i += 1) {
                    if (match_glob_parts(glob_parts, path_parts, next_glob_i, i)) return true;
                }
                return false;
            }

            if (path_i == path_parts.len) return false;
            if (std.mem.eql(u8, seg, "*") or std.mem.eql(u8, seg, path_parts[path_i])) {
                return match_glob_parts(glob_parts, path_parts, glob_i + 1, path_i + 1);
            }
            return false;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "from and extract round-trip" {
    const allocator = std.testing.allocator;

    const Spec = struct { dtype: u8, dim: i64 };
    const Params = struct { w: Spec, b: Spec };
    const Batch = struct { x: Spec };
    const Inputs = struct { params: Params, batch: Batch };

    const inputs = Inputs{
        .params = .{ .w = .{ .dtype = 1, .dim = 784 }, .b = .{ .dtype = 1, .dim = 128 } },
        .batch = .{ .x = .{ .dtype = 1, .dim = 64 } },
    };

    var tree = try Tree(Spec).from(allocator, inputs);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 3), tree.len());
    try std.testing.expectEqual(@as(i64, 784), tree.leaves[0].dim);
    try std.testing.expectEqual(@as(i64, 128), tree.leaves[1].dim);
    try std.testing.expectEqual(@as(i64, 64), tree.leaves[2].dim);

    const recovered = try tree.extract(Inputs);
    try std.testing.expectEqual(@as(i64, 784), recovered.params.w.dim);
    try std.testing.expectEqual(@as(i64, 128), recovered.params.b.dim);
    try std.testing.expectEqual(@as(i64, 64), recovered.batch.x.dim);
}

test "extract handles comptime-typed tuple fields" {
    const allocator = std.testing.allocator;

    const Inner = struct { a: i32, b: i32 };
    const c1: Inner = .{ .a = 1, .b = 2 };
    const c2: Inner = .{ .a = 3, .b = 4 };
    const comptime_tuple = .{ c1, c2 };

    var tree = try Tree(i32).from(allocator, comptime_tuple);
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 4), tree.len());

    const recovered = try tree.extract(@TypeOf(comptime_tuple));
    try std.testing.expectEqual(@as(i32, 1), recovered.@"0".a);
    try std.testing.expectEqual(@as(i32, 2), recovered.@"0".b);
    try std.testing.expectEqual(@as(i32, 3), recovered.@"1".a);
    try std.testing.expectEqual(@as(i32, 4), recovered.@"1".b);
}

test "map transforms leaf type" {
    const allocator = std.testing.allocator;

    const Inner = struct { val: i32 };
    const S = struct { a: Inner, b: Inner };

    var tree = try Tree(Inner).from(allocator, S{ .a = .{ .val = 10 }, .b = .{ .val = 20 } });
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
    var tree = try Tree(i32).from(allocator, S{ .a = 5, .b = 10 });
    defer tree.deinit();

    var mapped = try tree.map(i32, @as(i32, 3), struct {
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
    const Model = struct { embed: i32, layers: [3]Layer };

    var tree = try Tree(i32).from(allocator, Model{
        .embed = 100,
        .layers = .{
            .{ .w = 1, .b = 2 },
            .{ .w = 3, .b = 4 },
            .{ .w = 5, .b = 6 },
        },
    });
    defer tree.deinit();

    try std.testing.expectEqual(@as(usize, 7), tree.len());
    try std.testing.expectEqual(@as(i32, 100), tree.leaves[0]);

    const recovered = try tree.extract(Model);
    try std.testing.expectEqual(@as(i32, 100), recovered.embed);
    try std.testing.expectEqual(@as(i32, 3), recovered.layers[1].w);
    try std.testing.expectEqual(@as(i32, 6), recovered.layers[2].b);
}

test "contains, get, and get_const" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };
    var tree = try Tree(i32).from(allocator, S{ .a = 42, .b = 99 });
    defer tree.deinit();

    try std.testing.expect(tree.contains("a"));
    try std.testing.expect(!tree.contains("c"));
    try std.testing.expectEqual(@as(i32, 42), try tree.get("a"));
    try std.testing.expectEqual(@as(i32, 99), try tree.get("b"));
    try std.testing.expectError(error.PathNotFound, tree.get("c"));
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

test "from_slices copies inputs" {
    const allocator = std.testing.allocator;

    var leaves = [_]i32{ 1, 2 };
    const paths = [_][]const u8{ "x", "y" };

    var tree = try Tree(i32).from_slices(allocator, &leaves, &paths);
    defer tree.deinit();

    leaves[0] = 999;
    try std.testing.expectEqual(@as(i32, 1), tree.leaves[0]);
}

test "subtree view with relative paths" {
    const allocator = std.testing.allocator;

    const Mlp = struct { up_proj: i32, down_proj: i32 };
    const Layer = struct { mlp: Mlp, attn: i32 };
    const Model = struct { layers: [2]Layer };

    var tree = try Tree(i32).from(allocator, Model{
        .layers = .{
            .{ .mlp = .{ .up_proj = 10, .down_proj = 11 }, .attn = 12 },
            .{ .mlp = .{ .up_proj = 20, .down_proj = 21 }, .attn = 22 },
        },
    });
    defer tree.deinit();

    var v = try tree.subtree("layers.1.mlp");
    defer v.deinit();

    try std.testing.expectEqual(@as(usize, 2), v.len());
    try std.testing.expectEqual(@as(i32, 20), try v.get("up_proj"));
    try std.testing.expectEqual(@as(i32, 21), try v.get("down_proj"));
    try std.testing.expectError(error.PathNotFound, v.get("layers.1.mlp.up_proj"));
}

test "subtree_glob selects matching leaves" {
    const allocator = std.testing.allocator;

    const Layer = struct { mlp: struct { down_proj: i32, up_proj: i32 } };
    const Model = struct { layers: [2]Layer, norm: i32 };

    var tree = try Tree(i32).from(allocator, Model{
        .layers = .{
            .{ .mlp = .{ .down_proj = 10, .up_proj = 11 } },
            .{ .mlp = .{ .down_proj = 20, .up_proj = 21 } },
        },
        .norm = 30,
    });
    defer tree.deinit();

    var sub = try tree.subtree_glob("layers.**.up_proj");
    defer sub.deinit();

    try std.testing.expectEqual(@as(usize, 2), sub.len());
}

test "for_each_glob applies wildcard matches" {
    const allocator = std.testing.allocator;

    const Layer = struct {
        mlp: struct { down_proj: i32, up_proj: i32 },
        attn: i32,
    };
    const Model = struct { layers: [3]Layer, tail: i32 };

    var tree = try Tree(i32).from(allocator, Model{
        .layers = .{
            .{ .mlp = .{ .down_proj = 10, .up_proj = 11 }, .attn = 12 },
            .{ .mlp = .{ .down_proj = 20, .up_proj = 21 }, .attn = 22 },
            .{ .mlp = .{ .down_proj = 30, .up_proj = 31 }, .attn = 32 },
        },
        .tail = 99,
    });
    defer tree.deinit();

    try tree.for_each_glob("layers.*.mlp.down_proj", {}, struct {
        fn f(_: void, _: []const u8, leaf: *i32) void {
            leaf.* *= 10;
        }
    }.f);

    try std.testing.expectEqual(@as(i32, 100), try tree.get("layers.0.mlp.down_proj"));
    try std.testing.expectEqual(@as(i32, 200), try tree.get("layers.1.mlp.down_proj"));
    try std.testing.expectEqual(@as(i32, 300), try tree.get("layers.2.mlp.down_proj"));
    try std.testing.expectEqual(@as(i32, 11), try tree.get("layers.0.mlp.up_proj"));
    try std.testing.expectEqual(@as(i32, 99), try tree.get("tail"));
}

test "glob validation rejects partial wildcard segments" {
    const allocator = std.testing.allocator;

    const S = struct { a: i32, b: i32 };
    var tree = try Tree(i32).from(allocator, S{ .a = 1, .b = 2 });
    defer tree.deinit();

    try std.testing.expectError(error.InvalidGlobPattern, tree.for_each_glob("a*", {}, struct {
        fn f(_: void, _: []const u8, _: *i32) void {}
    }.f));
    try std.testing.expectError(error.InvalidGlobPattern, tree.subtree_glob("layers.**x"));
}

test "deinit_with frees leaf resources" {
    const allocator = std.testing.allocator;

    const Boxed = struct { val: []i32 };
    var tree = try Tree(Boxed).from(allocator, struct {
        a: Boxed,
        b: Boxed,
    }{
        .a = .{ .val = try allocator.dupe(i32, &.{ 1, 2 }) },
        .b = .{ .val = try allocator.dupe(i32, &.{ 3, 4 }) },
    });
    tree.deinit_with(struct {
        fn f(leaf: *Boxed) void {
            std.testing.allocator.free(leaf.val);
        }
    }.f);
}

test "render ascii tree" {
    const allocator = std.testing.allocator;

    const Model = struct {
        a: i32,
        b: struct { c: i32, d: i32 },
    };

    var tree = try Tree(i32).from(allocator, Model{ .a = 1, .b = .{ .c = 2, .d = 3 } });
    defer tree.deinit();

    var out = std.Io.Writer.Allocating.init(allocator);
    defer out.deinit();

    try tree.render(&out.writer, .{ .symbols = .ascii });
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
