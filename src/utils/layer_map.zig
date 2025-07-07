const std = @import("std");
const Allocator = std.mem.Allocator;
const NDTensor = @import("../ndtensor.zig").NDTensor;
const rtti = @import("../utils/rtti.zig");
const ClosurePointer = rtti.ClosurePointer;
const TypeID = rtti.TypeID;
const ArenaUnmanaged = @import("../allocators.zig").ArenaUnmanaged;
const TensorOpts = @import("../zigrad.zig").TensorOpts;

const stz = @import("../zigrad.zig").stz;
const zg = @import("../zigrad.zig");

const Self = @This();
const ParamMap = std.StringArrayHashMapUnmanaged(ClosurePointer);

const PopulateOpts = struct {
    shared: bool = true,
};

allocator: Allocator,
arena: ArenaUnmanaged,
map: ParamMap,

pub fn init(allocator: Allocator) Self {
    return .{ .allocator = allocator, .arena = .empty, .map = .empty };
}

pub fn deinit(self: *Self) void {
    for (self.map.values()) |*v|
        v.deinit();

    self.arena.deinit(self.allocator);
    self.map.deinit(self.allocator);
    self.* = undefined;
}

pub const PutOpts = struct {
    owned: bool = false,
};

pub fn put(self: *Self, key: []const u8, ptr: anytype, opts: PutOpts) !void {
    return self.put_closure(key, ClosurePointer.init(ptr, opts.owned));
}

pub fn put_closure(self: *Self, key: []const u8, ptr: ClosurePointer) !void {
    const _key = try self.arena.dupe(self.allocator, u8, key);
    errdefer self.arena.free(_key);
    try self.map.put(self.allocator, _key, ptr);
}

pub fn extract(
    self: *Self,
    ParamType: type,
    root: []const u8,
    opts: PopulateOpts,
) ParamType {
    var tmp: ParamType = undefined;
    self.populate(&tmp, root, opts);
    return tmp;
}

pub fn extract_map(
    self: *Self,
    ParamType: type,
    prefix: []const u8,
    opts: PopulateOpts,
) !std.StringArrayHashMapUnmanaged(ParamType) {
    var strings: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer strings.deinit(self.allocator);

    for (self.map.keys()) |key| {
        if (!std.mem.startsWith(u8, key, prefix)) continue;
        const end = std.mem.indexOfScalarPos(u8, key, prefix.len, '.') orelse key.len;
        try strings.put(self.allocator, key[0..end], {});
    }

    var output: std.StringArrayHashMapUnmanaged(ParamType) = .empty;
    errdefer output.deinit(self.allocator);

    for (strings.keys()) |key| {
        const param = extract(ParamType, self.map, key, opts);
        try output.put(self.allocator, key, param);
    }
    return output;
}

pub fn for_all_type(self: *Self, callable: anytype) void {
    const T = @TypeOf(callable);
    const U = if (@typeInfo(T) == .pointer) std.meta.Child(T) else T;
    std.debug.assert(@typeInfo(U) == .@"struct");
    std.debug.assert(@hasDecl(U, "call"));

    const params = @typeInfo(@TypeOf(U.call)).@"fn".params;
    std.debug.assert(params.len == 3);

    // argument type must be a pointer
    const arg_type = std.meta.Child(params[2].type orelse unreachable);

    var iter = self.map.iterator();
    while (iter.next()) |entry| {
        if (TypeID.init(arg_type) == entry.value_ptr.type_id) {
            callable.call(entry.key_ptr.*, entry.value_ptr.cast(arg_type));
        }
    }
}

pub fn for_all(self: *Self, callable: anytype) void {
    const T = @TypeOf(callable);
    const U = if (@typeInfo(T) == .pointer) std.meta.Child(T) else T;
    std.debug.assert(@typeInfo(U) == .@"struct");
    std.debug.assert(@hasDecl(U, "call"));

    const params = @typeInfo(@TypeOf(U.call)).@"fn".params;
    std.debug.assert(params.len == 3);

    // argument type must be a "anytype"
    std.debug.assert(params[2].type == null);

    const recognized: []const type = &.{
        NDTensor(f32),
        NDTensor(f64),
    };

    var iter = self.map.iterator();
    loop: while (iter.next()) |entry| {
        const k = entry.key_ptr;
        const v = entry.value_ptr;

        inline for (recognized) |t| {
            if (v.type_id == TypeID.init(t)) {
                callable.call(k.*, v.cast(t));
                continue :loop;
            }
        } else {
            @panic("Unimplemented");
        }
    }
}

const PathBuffer = std.BoundedArray(u8, 1024);

pub fn populate(
    self: *Self,
    ptr: anytype,
    root: []const u8,
    opts: PopulateOpts,
) void {
    var buf = PathBuffer.fromSlice(root) catch unreachable;
    recursive_populate(ptr, &self.map, &buf, opts.shared);
}

fn recursive_populate(
    ptr: anytype,
    map: *ParamMap,
    buf: *PathBuffer,
    shared: bool,
) void {
    const T = @TypeOf(ptr.*);

    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.fields) |field| {
            const ext = if (buf.len > 0)
                "." ++ field.name
            else
                field.name;

            buf.appendSlice(ext) catch unreachable;
            recursive_populate(&@field(ptr.*, field.name), map, buf);
            buf.len -= ext.len;
        },
        else => {
            const entry = map.get(buf.slice()) orelse unreachable;
            ptr.* = entry.cast(T);
            if (!shared) {
                _ = map.orderedRemove(buf.slice());
            }
        },
    }
}

// Print tree structure
//pub fn print_tree(self: *Self, prefix: []const u8) void {
//    switch (self.data) {
//        .subtree => |*subtrees| {
//            var iter = subtrees.iterator();
//            while (iter.next()) |entry| {
//                std.debug.print("{s}{s}/ (subtree)\n", .{ prefix, entry.key_ptr.* });
//                const new_prefix = std.fmt.allocPrint(self.allocator, "{s}  ", .{prefix}) catch return;
//                defer self.allocator.free(new_prefix);
//                entry.value_ptr.*.print_tree(new_prefix);
//            }
//        },
//        .leaf => |*leaves| {
//            var iter = leaves.iterator();
//            while (iter.next()) |entry| {
//                std.debug.print("{s}{s} (leaf: {d} elements)\n", .{ prefix, entry.key_ptr.*, entry.value_ptr.*.get_data().len });
//            }
//        },
//    }
//}
//
// Serialize the parameter tree to safetensors format
//pub fn serialize(self: *Self, allocator: std.mem.Allocator) ![]u8 {
//    var tensor_list = std.ArrayList(stz.Tensor).init(allocator);
//
//    const TensorCollector = struct {
//        tensor_list: *std.ArrayList(stz.Tensor),
//        allocator: std.mem.Allocator,
//
//        pub fn visit(_self: *@This(), path: []const u8, tensor: *T) !void {
//            const owned_path = try _self.allocator.dupe(u8, path);
//            errdefer _self.allocator.free(owned_path);
//
//            const tensor_data = tensor.get_data();
//            const shape_slice = tensor.get_shape();
//            // const owned_shape = _self.allocator.dupe(usize, shape_slice);
//            // errdefer _self.allocator.free(owned_shape);
//
//            if (!@hasDecl(T, "ValueType")) @compileError("Unable to infer tensor element type. Missing ValueType field.");
//            const dtype = switch (T.ValueType) {
//                f32 => stz.Dtype.f32,
//                f64 => stz.Dtype.f64,
//                i32 => stz.Dtype.i32,
//                i64 => stz.Dtype.i64,
//                u32 => stz.Dtype.u32,
//                u64 => stz.Dtype.u64,
//                else => stz.Dtype.f32,
//            };
//
//            const data_bytes = std.mem.sliceAsBytes(tensor_data);
//
//            try _self.tensor_list.append(stz.Tensor{
//                .name = owned_path,
//                .dtype = dtype,
//                // .shape = owned_shape,
//                .shape = shape_slice,
//                .data = @alignCast(data_bytes),
//            });
//        }
//    };
//
//    var collector = TensorCollector{
//        .tensor_list = &tensor_list,
//        .allocator = allocator,
//    };
//
//    try self.for_each(&collector);
//    return try stz.serialize_tensors(tensor_list, allocator);
//}

// Deserialize a parameter tree from safetensors format
// Leaf labels must be allocated and owned by caller. Use an arena, stack buf might be fine.
//pub fn deserialize(
//    data: []const u8,
//    allocator: Allocator,
//    device: zg.DeviceReference,
//    graph: *zg.Graph,
//) !Self {
//    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
//    defer st_file.deinit();
//
//    const tree = Self.init(allocator);
//    errdefer tree.deinit();
//
//    for (st_file.tensors) |tensor_info| {
//        const tensor_view = try st_file.get(tensor_info.name);
//        const ndtensor = try create_tensor_from_view(tensor_view, device, graph);
//        try tree.put(tensor_info.name, ndtensor);
//    }
//
//    std.debug.print("[deserialize] loaded:\n", .{});
//    tree.print_tree("");
//    std.debug.print("\n", .{});
//
//    return tree;
//}

fn closure_tensor(
    T: type,
    device: zg.DeviceReference,
    bytes: []const u8,
    shape: []const usize,
    opts: zg.TensorOpts,
) !*NDTensor(T) {
    return ClosurePointer.init(
        try NDTensor(T).from_slice(device, std.mem.sliceAsBytes(bytes), shape, opts),
    );
}

// Helper function to create NDTensor from TensorView
//fn create_tensor_from_view(
//    view: stz.TensorView,
//    device: zg.DeviceReference,
//    graph: *zg.Graph,
//) !ClosurePointer {
//    const opts = TensorOpts{
//        .requires_grad = true,
//        .graph = graph,
//    };
//
//    return switch (view.info.dtype) {
//        .u8 => view.data,
//        .bool => closure_tensor(bool, device, view.data, view.info.shape, opts),
//        .i8 => closure_tensor(i8, device, view.data, view.info.shape, opts),
//        .i16 => closure_tensor(i16, device, view.data, view.info.shape, opts),
//        .u16 => closure_tensor(u16, device, view.data, view.info.shape, opts),
//        .u32 => closure_tensor(u32, device, view.data, view.info.shape, opts),
//        .u64 => closure_tensor(u64, device, view.data, view.info.shape, opts),
//        .f16 => closure_tensor(f16, device, view.data, view.info.shape, opts),
//        .f32 => closure_tensor(f32, device, view.data, view.info.shape, opts),
//        .f64 => closure_tensor(f64, device, view.data, view.info.shape, opts),
//        .i32 => closure_tensor(i32, device, view.data, view.info.shape, opts),
//        .i64 => closure_tensor(i64, device, view.data, view.info.shape, opts),
//        else => return error.UnsupportedDtype,
//    };
//}

// Save parameter tree to file
//pub fn save_to_file(
//    self: *Self,
//    file_path: []const u8,
//    allocator: std.mem.Allocator,
//) !void {
//    const serialized_data = try self.serialize(allocator);
//    defer allocator.free(serialized_data);
//
//    try std.fs.cwd().writeFile(.{
//        .sub_path = file_path,
//        .data = serialized_data,
//        .flags = .{},
//    });
//}

// Load parameter tree from file
//pub fn load_from_file(
//    file_path: []const u8,
//    allocator: std.mem.Allocator,
//    device: anytype,
//    graph: anytype,
//) !*Self {
//    const file_data = try std.fs.cwd().readFileAlloc(allocator, file_path, .{});
//    defer allocator.free(file_data);
//
//    return Self.deserialize(file_data, allocator, device, graph);
//}
