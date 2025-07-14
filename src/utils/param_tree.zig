const std = @import("std");
const stz = @import("zigrad").stz;

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

        /// Apply visitor to all leaf parameters
        /// The vistor must have a visit method: visit(self, path: []const u8, tensor: *T) !void
        pub fn for_each(self: *Self, visitor: anytype) !void {
            const CT = @TypeOf(visitor);
            const VisitorType = switch (@typeInfo(CT)) {
                .pointer => std.meta.Child(CT),
                inline else => CT,
                // std.builtin.Type.Pointer => @compileLog("Foo."),
                // inline else => |CT| @compileLog("CT is: " ++ @typeName(CT)),
            };

            if (!@hasDecl(VisitorType, "visit")) {
                @compileError("Visitor must have a 'visit' method with signature: visit(self, path: []const u8, tensor: *T) !void");
            }

            var path_parts = std.ArrayList([]const u8).init(self.allocator);
            defer path_parts.deinit();
            try self.for_each_recursive(&path_parts, visitor);
        }

        fn for_each_recursive(self: *Self, path_parts: *std.ArrayList([]const u8), callback_obj: anytype) !void {
            switch (self.data) {
                .subtree => |*subtrees| {
                    var iter = subtrees.iterator();
                    while (iter.next()) |entry| {
                        try path_parts.append(entry.key_ptr.*);
                        defer _ = path_parts.pop();

                        try entry.value_ptr.*.for_each_recursive(path_parts, callback_obj);
                    }
                },
                .leaf => |*leaves| {
                    var iter = leaves.iterator();
                    while (iter.next()) |entry| {
                        try path_parts.append(entry.key_ptr.*);
                        defer _ = path_parts.pop();

                        const path_str = try self.build_path_string(path_parts.items);
                        defer self.allocator.free(path_str);
                        try callback_obj.visit(path_str, entry.value_ptr.*);
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
                        std.debug.print("{s}{s} (leaf: {d} elements)\n", .{ prefix, entry.key_ptr.*, entry.value_ptr.*.get_data().len });
                    }
                },
            }
        }

        /// Serialize the parameter tree to safetensors format
        pub fn serialize(self: *Self, allocator: std.mem.Allocator) ![]u8 {
            var tensor_list = std.ArrayList(stz.Tensor).init(allocator);
            defer tensor_list.deinit();
            defer for (tensor_list.items) |tensor| {
                allocator.free(tensor.name);
            };

            const TensorCollector = struct {
                tensor_list: *std.ArrayList(stz.Tensor),
                allocator: std.mem.Allocator,

                pub fn visit(_self: *@This(), path: []const u8, tensor: *T) !void {
                    const owned_path = try _self.allocator.dupe(u8, path);
                    errdefer _self.allocator.free(owned_path);

                    const tensor_data = tensor.get_data();
                    const shape_slice = tensor.get_shape();
                    // const owned_shape = _self.allocator.dupe(usize, shape_slice);
                    // errdefer _self.allocator.free(owned_shape);

                    if (!@hasDecl(T, "ValueType")) @compileError("Unable to infer tensor element type. Missing ValueType field.");
                    const dtype = switch (T.ValueType) {
                        f32 => stz.Dtype.f32,
                        f64 => stz.Dtype.f64,
                        i32 => stz.Dtype.i32,
                        i64 => stz.Dtype.i64,
                        u32 => stz.Dtype.u32,
                        u64 => stz.Dtype.u64,
                        else => stz.Dtype.f32,
                    };

                    const data_bytes = std.mem.sliceAsBytes(tensor_data);

                    try _self.tensor_list.append(stz.Tensor{
                        .name = owned_path,
                        .dtype = dtype,
                        // .shape = owned_shape,
                        .shape = shape_slice,
                        .data = @alignCast(data_bytes),
                    });
                }
            };

            var collector = TensorCollector{
                .tensor_list = &tensor_list,
                .allocator = allocator,
            };

            try self.for_each(&collector);
            return try stz.serialize_tensors(tensor_list, allocator);
        }

        /// Deserialize a parameter tree from safetensors format
        /// Leaf labels must be allocated and owned by caller. Use an arena, stack buf might be fine.
        pub fn deserialize(
            data: []const u8,
            allocator: std.mem.Allocator,
            label_allocator: std.mem.Allocator,
            device: anytype,
            graph: anytype,
        ) !*Self {
            var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
            defer st_file.deinit();

            const tree = try Self.create(allocator);
            errdefer tree.deinit();

            for (st_file.tensors) |tensor_info| {
                const tensor_view = try st_file.get(tensor_info.name);
                const ndtensor = try create_tensor_from_view(
                    tensor_view,
                    device,
                    graph,
                );
                try tree.put(try label_allocator.dupe(u8, tensor_info.name), ndtensor);
            }

            std.debug.print("[deserialize] loaded:\n", .{});
            tree.print_tree("");
            std.debug.print("\n", .{});
            return tree;
        }

        /// Helper function to create NDTensor from TensorView
        fn create_tensor_from_view(
            view: stz.TensorView,
            device: anytype,
            graph: anytype,
        ) !*T {
            const TensorOpts = @import("zigrad").TensorOpts;
            const tensor_opts = TensorOpts{
                .requires_grad = true,
                .graph = graph,
            };

            // NOTE: This assumes T is NDTensor(f32). we only support f32 for now
            const typed_data = switch (view.info.dtype) {
                .u8 => view.data,
                .bool => std.mem.bytesAsSlice(bool, view.data),
                .i8 => std.mem.bytesAsSlice(i8, view.data),
                .i16 => std.mem.bytesAsSlice(i16, view.data),
                .u16 => std.mem.bytesAsSlice(u16, view.data),
                .u32 => std.mem.bytesAsSlice(u32, view.data),
                .u64 => std.mem.bytesAsSlice(u64, view.data),
                .f16 => std.mem.bytesAsSlice(f16, view.data),
                .f32 => std.mem.bytesAsSlice(f32, view.data),
                .f64 => std.mem.bytesAsSlice(f64, view.data),
                .i32 => std.mem.bytesAsSlice(i32, view.data),
                .i64 => std.mem.bytesAsSlice(i64, view.data),
                else => return error.UnsupportedDtype,
            };
            return try T.from_slice(device, typed_data, view.info.shape, tensor_opts);
        }

        /// Save parameter tree to file
        pub fn save_to_file(
            self: *Self,
            file_path: []const u8,
            allocator: std.mem.Allocator,
        ) !void {
            const serialized_data = try self.serialize(allocator);
            defer allocator.free(serialized_data);

            const file = try std.fs.cwd().createFile(file_path, .{});
            defer file.close();

            try file.writeAll(serialized_data);
        }

        /// Load parameter tree from file
        pub fn load_from_file(
            file_path: []const u8,
            label_allocator: std.mem.Allocator,
            allocator: std.mem.Allocator,
            device: anytype,
            graph: anytype,
        ) !*Self {
            const file = try std.fs.cwd().openFile(file_path, .{});
            defer file.close();

            const file_size = try file.getEndPos();
            const file_data = try allocator.alloc(u8, file_size);
            defer allocator.free(file_data);

            _ = try file.readAll(file_data);

            return try Self.deserialize(file_data, allocator, label_allocator, device, graph);
        }
    };
}
