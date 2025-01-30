const std = @import("std");

// different than allocator error because
// device memory running out is not catastrophic
pub const Error = error{TensorOOM};

// NOTE: This is a *very* basic caching allocator to get
// device support off the ground. This is not meant to be
// a long standing object in the ecosystem. The configuration
// of this is fixed at the moment but will more accessible
// in future releases.

const c_allocator = std.heap.c_allocator;

pub fn CachingAllocator(Impl: type) type {
    return struct {
        const Self = @This();
        pub const Key = usize; // use slot length as key
        pub const List = std.SinglyLinkedList([*]u8);
        pub const Node = List.Node;
        pub const Map = std.AutoHashMapUnmanaged(Key, List);
        /// keeps track of freed allocations
        map: Map,
        /// used for allocating nodes for the free list
        node_buffer: []Node,
        /// free list for storing available nodes
        free_nodes: List,
        /// size difference of each allocation slot
        slot_bytes: usize,

        pub fn init(config: struct {
            slot_bytes: usize = 128 * @sizeOf(f32),
            node_count: usize = 400,
        }) Self {
            std.debug.assert(config.slot_bytes > 0);
            std.debug.assert(config.node_count > 0);

            const node_buffer = c_allocator.alloc(Node, config.node_count) catch {
                @panic("Unable to allocate node buffer for allocator");
            };

            for (1..config.node_count) |i|
                node_buffer[i - 1].next = &node_buffer[i];

            node_buffer[config.node_count - 1].next = null;

            var map: Map = .{};

            map.ensureTotalCapacity(c_allocator, 128) catch {
                @panic("Unable to allocate cache map for allocator");
            };

            return .{
                .map = map,
                .node_buffer = node_buffer,
                .free_nodes = .{ .first = &node_buffer[0] },
                .slot_bytes = config.slot_bytes,
            };
        }

        pub fn deinit(self: *Self, ctx: *anyopaque) void {
            self.clear(ctx);
            self.map.deinit(c_allocator);
            c_allocator.free(self.node_buffer);
            self.* = undefined;
        }

        pub fn clear(self: *Self, ctx: *anyopaque) void {
            var iter = self.map.valueIterator();
            while (iter.next()) |list| {
                while (list.popFirst()) |node| {
                    Impl.raw_free(node.data, ctx);
                    self.free_nodes.prepend(node);
                }
            }
        }

        pub fn alloc(self: *Self, T: type, len: usize, ctx: *anyopaque) Error![]T {
            const bytes = len * @sizeOf(T);
            const ptr: [*]u8 = blk: {
                break :blk self.alloc_impl(bytes, ctx) orelse {
                    self.clear(ctx);
                    break :blk self.alloc_impl(bytes, ctx) orelse {
                        return error.TensorOOM;
                    };
                };
            };
            return cast_slice(T, ptr, len);
        }

        fn alloc_impl(self: *Self, bytes: usize, ctx: *anyopaque) ?[*]u8 {
            const slot = get_slot(bytes, self.slot_bytes);
            blk: {
                const list = self.map.getPtr(slot) orelse break :blk;
                const node = list.popFirst() orelse break :blk;
                self.free_nodes.prepend(node);
                return @ptrCast(node.data);
            }
            return Impl.raw_alloc(slot, ctx);
        }

        pub fn free(self: *Self, buf: anytype, ctx: *anyopaque) void {
            return self.free_impl(std.mem.sliceAsBytes(@constCast(buf)), ctx);
        }

        pub fn free_impl(self: *Self, buf: []u8, ctx: *anyopaque) void {
            const slot = get_slot(buf.len, self.slot_bytes);

            // if we have no more free nodes, we're forced to release the memory
            const node: *Node = self.free_nodes.popFirst() orelse {
                return Impl.raw_free(buf.ptr, ctx);
            };
            // if we fail to allocate for the map, release the entry
            var result = self.map.getOrPut(c_allocator, slot) catch {
                return Impl.raw_free(buf.ptr, ctx);
            };

            if (!result.found_existing) {
                result.value_ptr.* = .{};
            }

            node.data = @ptrCast(buf.ptr);

            result.value_ptr.prepend(node);
        }
    };
}

inline fn get_slot(len: usize, slot_bytes: usize) usize {
    return ((len / slot_bytes) +| 1) *| slot_bytes;
}

inline fn cast_slice(T: type, ptr: [*]u8, len: usize) []T {
    const tptr: [*]T = @ptrCast(@alignCast(ptr));
    return tptr[0..len];
}
