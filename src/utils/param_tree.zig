const std = @import("std");

pub fn ParamTree(comptime T: type) type {
    return struct {
        const Self = @This();

        const Data = union(enum) {
            subtree: std.StringHashMap(*Self),
            leaf: std.StringHashMap(*T),
        };

        data: Data,
        allocator: std.mem.Allocator,

        /// Internal. Create subtree
        fn init_subtree(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .data = Data{ .subtree = std.StringHashMap(*Self).init(allocator) },
                .allocator = allocator,
            };
            return self;
        }

        /// Create a root tree
        pub fn create(allocator: std.mem.Allocator) !*Self {
            return try Self.init_subtree(allocator);
        }

        /// Recursive free
        pub fn deinit(self: *Self) void {
            switch (self.data) {
                .subtree => |*subtrees| {
                    var iter = subtrees.iterator();
                    while (iter.next()) |entry| {
                        entry.value_ptr.*.deinit();
                    }
                    subtrees.deinit();
                },
                .leaf => |*leaves| {
                    var iter = leaves.iterator();
                    while (iter.next()) |entry| {
                        entry.value_ptr.*.deinit();
                    }
                    leaves.deinit();
                },
            }
            self.allocator.destroy(self);
        }

        /// Get a parameter by path
        pub fn get(self: *Self, path: []const u8) ?*T {
            var current = self;
            var it = std.mem.tokenizeScalar(u8, path, '.');

            while (it.next()) |component| {
                if (it.peek()) |_| {
                    // More components remaining, traverse subtree
                    switch (current.data) {
                        .subtree => |*subtrees| {
                            current = subtrees.get(component) orelse return null;
                        },
                        .leaf => return null, // Cant traverse through a leaf
                    }
                } else {
                    // Last component, expect leaf
                    return switch (current.data) {
                        .leaf => |*leaves| leaves.get(component),
                        .subtree => null, // Expected leaf got subtree
                    };
                }
            }

            return null;
        }

        /// Store a parameter at path, creating intermediate nodes as needed
        pub fn put(self: *Self, path: []const u8, tensor: *T) !void {
            var current = self;
            var it = std.mem.tokenizeScalar(u8, path, '.');

            var final_component: ?[]const u8 = null;
            while (it.next()) |component| {
                if (it.peek()) |_| {
                    // More components - ensure we have subtree container and navigate
                    switch (current.data) {
                        .subtree => |*subtrees| {
                            const result = try subtrees.getOrPut(component);
                            // insert subtree node
                            if (!result.found_existing) {
                                result.value_ptr.* = try Self.init_subtree(current.allocator);
                            }
                            current = result.value_ptr.*;
                        },
                        .leaf => return error.PathConflict, // Cant traverse through leaf
                    }
                } else {
                    final_component = component;
                    break;
                }
            }

            if (final_component) |name| {
                // Convert to leaf container if needed
                switch (current.data) {
                    .subtree => |*subtrees| {
                        // Check if subtrees is empty - if so, we can convert to leaf container
                        if (subtrees.count() == 0) {
                            subtrees.deinit();
                            current.data = Data{ .leaf = std.StringHashMap(*T).init(current.allocator) };
                        } else {
                            return error.PathConflict; // Cant mix subtrees and leaves
                        }
                    },
                    .leaf => {}, // Already a leaf container, good
                }

                // Add leaf
                switch (current.data) {
                    .leaf => |*leaves| {
                        const result = try leaves.getOrPut(name);
                        if (result.found_existing) {
                            return error.DuplicateParameter;
                        }
                        result.value_ptr.* = tensor;
                    },
                    .subtree => unreachable,
                }
            } else {
                return error.InvalidPath;
            }
        }

        /// Apply function to all leaf parameters
        pub fn for_each(self: *Self, callback: fn (path: []const u8, tensor: *T) void) void {
            var path_parts = std.ArrayList([]const u8).init(self.allocator);
            defer path_parts.deinit();
            self.for_each_recursive(&path_parts, callback);
        }

        fn for_each_recursive(self: *Self, path_parts: *std.ArrayList([]const u8), callback: fn (path: []const u8, tensor: *T) void) void {
            switch (self.data) {
                .subtree => |*subtrees| {
                    var iter = subtrees.iterator();
                    while (iter.next()) |entry| {
                        path_parts.append(entry.key_ptr.*) catch return;
                        defer _ = path_parts.pop();

                        entry.value_ptr.*.for_each_recursive(path_parts, callback);
                    }
                },
                .leaf => |*leaves| {
                    var iter = leaves.iterator();
                    while (iter.next()) |entry| {
                        path_parts.append(entry.key_ptr.*) catch return;
                        defer _ = path_parts.pop();

                        const path_str = self.build_path_string(path_parts.items) catch return;
                        defer self.allocator.free(path_str);
                        callback(path_str, entry.value_ptr.*);
                    }
                },
            }
        }

        fn build_path_string(self: *Self, parts: []const []const u8) ![]u8 {
            if (parts.len == 0) return try self.allocator.dupe(u8, "");

            var total_len: usize = 0;
            for (parts) |part| {
                total_len += part.len;
            }
            total_len += parts.len - 1; // dots between parts

            var result = try self.allocator.alloc(u8, total_len);
            var pos: usize = 0;

            for (parts, 0..) |part, i| {
                if (i > 0) {
                    result[pos] = '.';
                    pos += 1;
                }
                @memcpy(result[pos .. pos + part.len], part);
                pos += part.len;
            }

            return result;
        }

        /// Print tree structure
        pub fn print_tree(self: *Self, prefix: []const u8) void {
            switch (self.data) {
                .subtree => |*subtrees| {
                    var iter = subtrees.iterator();
                    while (iter.next()) |entry| {
                        std.debug.print("{s}{s}/ (subtree)\n", .{ prefix, entry.key_ptr.* });
                        const new_prefix = std.fmt.allocPrint(self.allocator, "{s}  ", .{prefix}) catch return;
                        defer self.allocator.free(new_prefix);
                        entry.value_ptr.*.print_tree(new_prefix);
                    }
                },
                .leaf => |*leaves| {
                    var iter = leaves.iterator();
                    while (iter.next()) |entry| {
                        std.debug.print("{s}{s} (leaf: {} elements)\n", .{ prefix, entry.key_ptr.*, entry.value_ptr.*.get_data().len });
                    }
                },
            }
        }
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const zg = @import("zigrad");
    const NDTensor = zg.NDTensor;
    const Graph = zg.Graph;
    const device = zg.device;
    var cpu = device.HostDevice.init();
    defer cpu.deinit();
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    // Create param tree
    const tree = try ParamTree(NDTensor(f32)).create(allocator);
    defer tree.deinit();

    // Create test tensors
    const weight1 = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{ 1.0, 2.0, 3.0 },
        &.{3},
        .{ .requires_grad = true },
    );
    const bias1 = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{ 0.1, 0.2 },
        &.{2},
        .{ .requires_grad = true },
    );
    const weight2 = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{ 4.0, 5.0, 6.0, 7.0 },
        &.{4},
        .{ .requires_grad = true },
    );
    const single_param = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{42.0},
        &.{1},
        .{ .requires_grad = true },
    );

    std.debug.print("=== Testing Final ParamTree ===\n", .{});

    // Test putting parameters
    try tree.put("layer1.weight", weight1);
    try tree.put("layer1.bias", bias1);
    try tree.put("layer2.weight", weight2);

    std.debug.print("[✓] Added nested parameters\n", .{});

    // Test single parameter conflict - this should fail because root already has subtrees
    if (tree.put("single", single_param)) {
        std.debug.print("[✗] Should have failed on path conflict (mixing subtrees and leaves at root)\n", .{});
        // Clean up if it somehow succeeded
        single_param.deinit();
    } else |err| {
        if (err == error.PathConflict) {
            std.debug.print("[✓] Correctly detected path conflict (can't mix subtrees and leaves)\n", .{});
        } else {
            std.debug.print("[✗] Got wrong error: {}\n", .{err});
        }
        single_param.deinit();
    }

    // Test getting parameters
    if (tree.get("layer1.weight")) |w| {
        std.debug.print("[✓] Retrieved layer1.weight: {} elements\n", .{w.get_data().len});
    } else {
        std.debug.print("[✗] Failed to retrieve layer1.weight\n", .{});
    }

    // Test non-existent path
    if (tree.get("nonexistent.path")) |_| {
        std.debug.print("[✗] Should not have found nonexistent.path\n", .{});
    } else {
        std.debug.print("[✓] Correctly returned null for nonexistent path\n", .{});
    }

    // Test duplicate parameter error
    const duplicate_weight = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{ 8, 9 },
        &.{2},
        .{ .requires_grad = true },
    );
    defer duplicate_weight.deinit();

    if (tree.put("layer1.weight", duplicate_weight)) {
        std.debug.print("[✗] Should have failed on duplicate parameter\n", .{});
    } else |err| {
        if (err == error.DuplicateParameter) {
            std.debug.print("[✓] Correctly detected duplicate parameter\n", .{});
        } else {
            std.debug.print("[✗] Got wrong error: {}\n", .{err});
        }
    }

    // Test path conflict
    const conflict_tensor = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{10},
        &.{1},
        .{ .requires_grad = true },
    );

    defer conflict_tensor.deinit();

    if (tree.put("layer1.weight.sublayer", conflict_tensor)) {
        std.debug.print("[✗] Should have failed on path conflict\n", .{});
    } else |err| {
        if (err == error.PathConflict) {
            std.debug.print("[✓] Correctly detected path conflict\n", .{});
        } else {
            std.debug.print("[✗] Got wrong error: {}\n", .{err});
        }
    }

    // Test for_each
    std.debug.print("\n=== Testing for_each ===\n", .{});
    const print_param = struct {
        fn callback(path: []const u8, tensor: *NDTensor(f32)) void {
            std.debug.print("Parameter: {s} -> {} elements\n", .{ path, tensor.get_data().len });
        }
    }.callback;

    tree.for_each(print_param);

    // Print tree structure
    std.debug.print("\n=== Tree Structure ===\n", .{});
    tree.print_tree("");

    // Test single parameter tree (create a separate tree for this)
    std.debug.print("\n=== Testing Single Parameter Tree ===\n", .{});
    const single_tree = try ParamTree(NDTensor(f32)).create(allocator);
    defer single_tree.deinit();

    const single_param2 = try NDTensor(f32).from_slice(
        &graph,
        cpu.reference(),
        &[_]f32{99},
        &.{1},
        .{ .requires_grad = true },
    );

    try single_tree.put("single_param", single_param2);
    std.debug.print("[✓] Added single parameter to empty tree\n", .{});

    if (single_tree.get("single_param")) |s| {
        std.debug.print("[✓] Retrieved single parameter: {}\n", .{s.get(0)});
    }

    std.debug.print("Single tree structure:\n", .{});
    single_tree.print_tree("");

    std.debug.print("\n=== Tests completed ===\n", .{});
}
