//! Hierarchical parameter storage. Stores tensors in a key-value map using
//! dot-separated paths (e.g., "encoder.layer1.weights") and can perform
//! strongly typed extraction through type introspection and comptime
//! metaprogramming.
//!
//! Supports serialization to SafeTensors format.
//!
//! Example usage:
//! ```zig
//! var map = LayerMap.init(allocator);
//! try map.put("model.weights", tensor, .{});
//! const MyModel = struct { model: struct { weights: NDTensor(f32) } };
//! const params = map.extract(MyModel, "", .{});
//! ```
const std = @import("std");
const Allocator = std.mem.Allocator;
const NDTensor = @import("../ndtensor.zig").NDTensor;
const rtti = @import("../utils/rtti.zig");
const ClosurePointer = rtti.ClosurePointer;
const TypeID = rtti.TypeID;
const ArenaUnmanaged = @import("../allocators.zig").ArenaUnmanaged;
const TensorOpts = @import("../zigrad.zig").TensorOpts;

const Self = @This();
const stz = @import("../zigrad.zig").stz;
const zg = @import("../zigrad.zig");

/// Stores type erased values
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
/// Backing storage for entries as type erased values
map: ParamMap,

pub fn init(allocator: Allocator) Self {
    return .{ .allocator = allocator, .arena = .empty, .map = .empty };
}

pub fn deinit(self: *Self) void {
    for (self.map.values()) |*v| v.deinit();

    self.arena.deinit(self.allocator);
    self.map.deinit(self.allocator);
    self.* = undefined;
}

pub const PutOpts = struct {
    owned: bool = false,
};

/// Stores a value in the map at the specified key path.
pub fn put(
    self: *Self,
    /// Hierarchical key path (e.g., "layer.weights")
    key: []const u8,
    /// Pointer to value to store (e.g., a `*zg.NDTensor`)
    ptr: anytype,
    /// Control ownership behavior
    opts: PutOpts,
) !void {
    try self.put_closure(key, ClosurePointer.init(ptr, opts.owned));
}

/// Lower-level version of `put()` operating on type-erased values.
pub fn put_closure(
    self: *Self,
    /// Hierarchical key path
    key: []const u8,
    /// Type-erased pointer to store
    ptr: ClosurePointer,
) !void {
    const _key = try self.arena.dupe(self.allocator, u8, key);
    errdefer self.arena.free(_key);
    try self.map.put(self.allocator, _key, ptr);
}

/// Extracts items from the layer map by populating the provided struct type
/// with entries from the map based on field names. Recurses on the fields of
/// `ParamType` until one of `SUPPORTED_TYPES` is found.
///
/// Field paths are built with "." as the delimeter (e.g., "layer.weights").
pub fn extract(
    self: *Self,
    /// The struct type to populate from the map
    ParamType: type,
    /// The root path prefix for all field lookups in the map
    root: []const u8,
    /// Configure ownership during population
    opts: PopulateOpts,
) ParamType {
    var tmp: ParamType = undefined;
    self.populate(&tmp, root, opts);
    return tmp;
}

/// Extracts multiple parameter groups from the map based on shared prefixes.
/// Returns a hashmap where keys are the discovered prefixes and values are
/// the populated parameter structures.
///
/// For example, given keys "model1.weights" and "model2.weights" and using
/// prefix "model" returns map with "model1" and "model2" entries.
pub fn extract_map(
    self: *Self,
    /// The struct type to populate for each group
    ParamType: type,
    /// The common prefix to search for
    prefix: []const u8,
    /// Configure ownership during population
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

/// Visits entries matching a specific type specified by signature of the vistor's `visit()` method.
///
/// Example visitor:
/// ```zig
/// struct {
///     pub fn visit(_: @This(), key: []const u8, entry: *zg.NDTensor(f32)) void {
///         // ...
///     }
/// }
/// ```
pub fn for_each_type(
    self: *Self,
    /// Visitor with `visit(self: @This(), key: []const u8, entry: *SomeCustomType)` method
    visitor: anytype,
) void {
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

/// Applies a visitor to every leaf.
///
/// A valid visitor must have a valid `visit` method.
///
/// Example visitor:
/// ```zig
/// struct {
///     pub fn visit(_: @This(), key: []const u8, entry: anytype) void {
///         // ...
///     }
/// }
/// ```
pub fn for_each(
    self: *Self,
    /// Visitor with `visit(self: @This(), key: []const u8, entry: anytype)` method
    visitor: anytype,
) void {
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
                std.debug.print("{s}{s}{s}{s}\n", .{
                    prefix[0..stack.len],
                    "\u{2514}", // └
                    "\u{2500}", // ─
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
                prefix[0 .. stack.len + 2],
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

/// Populates an existing struct with entries from the map.
/// Lower-level function used by `extract()`.
///
/// E.g., `var params: MyStruct = undefined; populate(&params, "root", .{});`
pub fn populate(
    self: *Self,
    /// Struct instance to populate
    ptr: anytype,
    /// Root path prefix for lookups
    root: []const u8,
    /// Configure ownership during population
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

    // Check if this is a supported tensor type first
    inline for (SUPPORTED_TYPES) |supported_type| {
        if (T == supported_type) {
            const entry = map.get(buf.slice()) orelse unreachable;
            ptr.* = entry.cast(T).*;
            if (!shared) _ = map.orderedRemove(buf.slice());
            return;
        }
    }

    // If not a supported tensor type recurse into struct fields
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.fields) |field| {
            const ext = if (buf.len > 0) "." ++ field.name else field.name;
            buf.appendSlice(ext) catch unreachable;
            recursive_populate(&@field(ptr.*, field.name), map, buf, shared);
            buf.len -= ext.len;
        },
        else => {
            @compileError("Unsupported leaf type in parameter structure" ++ @typeName(T));
        },
    }
}

/// Serializes param tensors to SafeTensors binary format.
/// Returns allocated byte buffer. COM.
pub fn serialize(
    self: *Self,
    /// Allocator passed to `stz.serialize_tensors()` (from `safetensors-zg`)
    allocator: Allocator,
) ![]u8 {
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

/// Creates `LayerMap` from SafeTensors binary data.
/// Tensors are allocated on the specified device.
pub fn deserialize(
    /// SafeTensors binary data
    data: []const u8,
    /// Allocator for new `LayerMap`
    allocator: Allocator,
    /// Target device for tensors
    device: zg.DeviceReference,
    /// Control tensor ownership and graph
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

/// Saves parameters to a SafeTensors file.
/// Convenience wrapper around `serialize()`.
pub fn save_to_file(
    self: *Self,
    /// Output file path
    file_path: []const u8,
    /// Allocator for temporary buffer
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

/// Loads parameters from a SafeTensors file.
/// Convenience wrapper around `deserialize()`.
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

test {
    const LayerMap = Self;
    const allocator = std.testing.allocator;

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var graph = zg.Graph.init(allocator, .{});
    defer graph.deinit();

    zg.global_graph_init(allocator, .{});
    defer zg.global_graph_deinit();

    const x = try zg.NDTensor(f32).random(cpu.reference(), &.{ 2, 6 }, .uniform, .{
        .label = "x: f32",
        .graph = &graph,
    });
    defer x.deinit();

    const y = try zg.NDTensor(f64).random(cpu.reference(), &.{ 2, 2 }, .uniform, .{
        .label = "y: f64",
        .graph = &graph,
    });
    defer y.deinit();

    var lmap = LayerMap.init(allocator);
    defer lmap.deinit();

    try lmap.put("layer_a.foo.bar.weights", x, .{ .owned = false });
    try lmap.put("layer_a.foo.bar.bias", y, .{ .owned = false });

    lmap.for_each(struct { // target every node in the graph (auto-cast)
        pub fn visit(_: @This(), key: []const u8, t: anytype) void {
            std.debug.print("LABEL (ALL): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});

    lmap.for_each_type(struct { // only target NDTensor(f32)
        pub fn visit(_: @This(), key: []const u8, t: *zg.NDTensor(f32)) void {
            std.debug.print("LABEL (f32): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});

    lmap.for_each_type(struct { // only target NDTensor(f64)
        pub fn visit(_: @This(), key: []const u8, t: *zg.NDTensor(f64)) void {
            std.debug.print("LABEL (f64): {?s}, key: {s}\n", .{ t.get_label(), key });
        }
    }{});

    var counter: struct {
        total_params: usize = 0,
        total_tensors: u32 = 0,
        largest_tensor: usize = 0,
        largest_tensor_key: []const u8 = "",
        pub fn visit(self: *@This(), key: []const u8, t: anytype) void {
            const param_count = t.get_size();
            self.total_tensors += 1;
            self.total_params += param_count;

            if (self.largest_tensor < param_count) {
                self.largest_tensor = param_count;
                self.largest_tensor_key = key;
            }
        }
    } = .{};

    lmap.for_each(&counter);
    std.debug.print(
        \\Parameter Statistics:
        \\  Total tensors: {d}
        \\  Largest tensor: {d} at '{s}'
        \\
    , .{
        counter.total_tensors,
        counter.largest_tensor,
        counter.largest_tensor_key,
    });

    lmap.print_tree();

    try lmap.save_to_file("here.stz", allocator);

    var tree = try LayerMap.load_from_file("here.stz", allocator, cpu.reference(), .{
        .owning = true,
    });
    defer tree.deinit();

    tree.print_tree();

    try std.fs.cwd().deleteFile("here.stz");

    // ----------------------------
    // Extract
    // ----------------------------
    const w = lmap.extract(zg.NDTensor(f32), "layer_a.foo.bar.weights", .{});
    std.debug.print("Extracted tensor: {s}\n", .{w.get_label().?});
    const LayerA = struct {
        foo: struct {
            bar: struct {
                weights: zg.NDTensor(f32),
                bias: zg.NDTensor(f64),
            },
        },
    };
    // TODO: proper error handling for extract(). try passing a bad prefix (e.g., "") to see what I mean.
    const la = lmap.extract(LayerA, "layer_a", .{});
    std.debug.print("Extracted struct: {}\n", .{la});
}
