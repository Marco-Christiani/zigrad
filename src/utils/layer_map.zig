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

const SUPPORTED_TYPES: []const type = &.{
    NDTensor(f32),
    NDTensor(f64),
};

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

pub fn for_each_type(self: *Self, visitor: anytype) void {
    const T = @TypeOf(visitor);
    const U = if (@typeInfo(T) == .pointer) std.meta.Child(T) else T;
    comptime std.debug.assert(@typeInfo(U) == .@"struct");
    comptime std.debug.assert(@hasDecl(U, "visit"));

    const params = @typeInfo(@TypeOf(U.visit)).@"fn".params;
    comptime std.debug.assert(params.len == 3);

    // argument type must be a pointer
    const arg_type = std.meta.Child(params[2].type orelse unreachable);

    var iter = self.map.iterator();
    while (iter.next()) |entry| {
        if (TypeID.init(arg_type) == entry.value_ptr.type_id) {
            visitor.visit(entry.key_ptr.*, entry.value_ptr.cast(arg_type));
        }
    }
}

pub fn for_each(self: *Self, visitor: anytype) void {
    const T = @TypeOf(visitor);
    const U = if (@typeInfo(T) == .pointer) std.meta.Child(T) else T;
    comptime std.debug.assert(@typeInfo(U) == .@"struct");
    comptime std.debug.assert(@hasDecl(U, "visit"));

    const params = @typeInfo(@TypeOf(U.visit)).@"fn".params;
    comptime std.debug.assert(params.len == 3);

    // argument type must be a "anytype"
    comptime std.debug.assert(params[2].type == null);

    var iter = self.map.iterator();
    loop: while (iter.next()) |entry| {
        const k = entry.key_ptr;
        const v = entry.value_ptr;

        inline for (SUPPORTED_TYPES) |t| {
            if (v.type_id == TypeID.init(t)) {
                visitor.visit(k.*, v.cast(t));
                continue :loop;
            }
        } else {
            @panic("Unimplemented");
        }
    }
}

/// Print tree structure
pub fn print_tree(self: *Self) void {
    const sorted_keys = self.allocator.dupe([]const u8, self.map.keys()) catch return;
    defer self.allocator.free(sorted_keys);

    std.sort.pdq([]const u8, sorted_keys, {}, struct {
        pub fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    const keys = sorted_keys;
    const prefix: [128]u8 = @splat(' ');

    var stack: std.BoundedArray(usize, 64) = .{};
    var key_idx: usize = 0;
    var key_pos: usize = 0;

    outer: while (key_idx < keys.len) {
        const key = keys[key_idx];

        while (std.mem.indexOfScalarPos(u8, key, key_pos, '.')) |sep_pos| {
            stack.append(sep_pos) catch unreachable;

            if (key_pos == 0)
                std.debug.print("{s}\n", .{key[key_pos..sep_pos]})
            else if (key_pos != sep_pos) {
                std.debug.print("{s}∟{s}\n", .{
                    prefix[0..stack.len],
                    key[key_pos..sep_pos],
                });
            }

            key_pos = sep_pos + 1;
        }

        while (key_idx < keys.len and std.mem.startsWith(
            u8,
            keys[key_idx],
            key[0..key_pos],
        )) : (key_idx += 1) {
            std.debug.print("{s}.{s}\n", .{
                prefix[0 .. stack.len + 1],
                keys[key_idx][key_pos..],
            });
        }

        if (key_idx == keys.len)
            return;

        while (stack.pop()) |sep_pos| {
            if (std.mem.startsWith(u8, keys[key_idx], key[0..sep_pos])) {
                key_pos = sep_pos;
                continue :outer;
            }
        } else {
            key_pos = 0;
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

/// Serialize the parameter tree to safetensors format
pub fn serialize(self: *Self, allocator: Allocator) ![]u8 {
    const TensorCollector = struct {
        tensor_list: std.ArrayList(stz.Tensor),

        pub fn visit(_self: *@This(), key: []const u8, tensor: anytype) void {
            const T = std.meta.Child(@TypeOf(tensor));

            if (!@hasDecl(T, "ValueType"))
                @compileError("Unable to infer tensor element type. Missing ValueType field.");

            const dtype = switch (T.ValueType) {
                f32 => stz.Dtype.f32,
                f64 => stz.Dtype.f64,
                i32 => stz.Dtype.i32,
                i64 => stz.Dtype.i64,
                u32 => stz.Dtype.u32,
                u64 => stz.Dtype.u64,
                else => stz.Dtype.f32,
            };

            const tensor_data = tensor.get_data();
            const data_bytes = std.mem.sliceAsBytes(tensor_data);
            _self.tensor_list.appendAssumeCapacity(.{
                .name = key,
                .dtype = dtype,
                .shape = tensor.get_shape(),
                .data = @alignCast(data_bytes),
            });
        }
    };

    var collector = TensorCollector{
        .tensor_list = try std.ArrayList(stz.Tensor).initCapacity(self.allocator, self.map.count()),
    };
    defer collector.tensor_list.deinit();

    self.for_each(&collector);

    return stz.serialize_tensors(collector.tensor_list, allocator);
}

pub fn deserialize(
    data: []const u8,
    allocator: Allocator,
    device: zg.DeviceReference,
    opts: LoadOpts,
) !Self {
    const graph = opts.graph orelse zg.global_graph_get();

    var st_file = try stz.SafeTensorsFile.deserialize(data, allocator);
    defer st_file.deinit();

    var tree = Self.init(allocator);
    errdefer tree.deinit();

    for (st_file.tensors) |tensor_info| {
        const tensor_view = try st_file.get(tensor_info.name);
        const ndtensor = try create_tensor_from_view(tensor_view, device, graph, opts.owning);
        try tree.put_closure(tensor_info.name, ndtensor);
    }
    return tree;
}

fn closure_tensor(
    T: type,
    device: zg.DeviceReference,
    bytes: []align(8) const u8,
    shape: []const usize,
    owned: bool,
    opts: zg.TensorOpts,
) !ClosurePointer {
    const slice: []const T = std.mem.bytesAsSlice(T, bytes);
    return ClosurePointer.init(try NDTensor(T).from_slice(device, slice, shape, opts), owned);
}

// Helper function to create NDTensor from TensorView
fn create_tensor_from_view(
    view: stz.TensorView,
    device: zg.DeviceReference,
    graph: *zg.Graph,
    owned: bool,
) !ClosurePointer {
    const opts = TensorOpts{
        .requires_grad = true,
        .graph = graph,
    };

    return switch (view.info.dtype) {
        .u8 => closure_tensor(u8, device, view.data, view.info.shape, owned, opts),
        .bool => closure_tensor(bool, device, view.data, view.info.shape, owned, opts),
        .i8 => closure_tensor(i8, device, view.data, view.info.shape, owned, opts),
        .i16 => closure_tensor(i16, device, view.data, view.info.shape, owned, opts),
        .u16 => closure_tensor(u16, device, view.data, view.info.shape, owned, opts),
        .u32 => closure_tensor(u32, device, view.data, view.info.shape, owned, opts),
        .u64 => closure_tensor(u64, device, view.data, view.info.shape, owned, opts),
        .f16 => closure_tensor(f16, device, view.data, view.info.shape, owned, opts),
        .f32 => closure_tensor(f32, device, view.data, view.info.shape, owned, opts),
        .f64 => closure_tensor(f64, device, view.data, view.info.shape, owned, opts),
        .i32 => closure_tensor(i32, device, view.data, view.info.shape, owned, opts),
        .i64 => closure_tensor(i64, device, view.data, view.info.shape, owned, opts),
        else => return error.UnsupportedDtype,
    };
}

// Save parameter tree to file
pub fn save_to_file(
    self: *Self,
    file_path: []const u8,
    allocator: Allocator,
) !void {
    const serialized_data = try self.serialize(allocator);
    defer self.allocator.free(serialized_data);

    try std.fs.cwd().writeFile(.{
        .sub_path = file_path,
        .data = serialized_data,
        .flags = .{},
    });
}

pub const LoadOpts = struct {
    owning: bool = true,
    graph: ?*zg.Graph = null,
};

// Load parameter tree from file
pub fn load_from_file(
    file_path: []const u8,
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    opts: LoadOpts,
) !Self {
    const file_data = try std.fs.cwd().readFileAlloc(allocator, file_path, std.math.maxInt(usize));
    defer allocator.free(file_data);

    return deserialize(file_data, allocator, device, opts);
}
