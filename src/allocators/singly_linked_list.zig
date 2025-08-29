//! Back-port of singly linked list from std
const std = @import("std");
const debug = std.debug;
const assert = debug.assert;
const testing = std.testing;

pub fn SinglyLinkedList(T: type) type {
    return struct {
        const Self = @This();

        first: ?*Node = null,

        /// This struct contains only a next pointer and not any data payload. The
        /// intended usage is to embed it intrusively into another data structure and
        /// access the data with `@fieldParentPtr`.
        pub const Node = struct {
            data: T,
            next: ?*Node = null,

            pub fn insertAfter(node: *Node, new_node: *Node) void {
                new_node.next = node.next;
                node.next = new_node;
            }

            /// Remove the node after the one provided, returning it.
            pub fn removeNext(node: *Node) ?*Node {
                const next_node = node.next orelse return null;
                node.next = next_node.next;
                return next_node;
            }

            /// Iterate over the singly-linked list from this node, until the final
            /// node is found.
            ///
            /// This operation is O(N). Instead of calling this function, consider
            /// using a different data structure.
            pub fn findLast(node: *Node) *Node {
                var it = node;
                while (true) {
                    it = it.next orelse return it;
                }
            }

            /// Iterate over each next node, returning the count of all nodes except
            /// the starting one.
            ///
            /// This operation is O(N). Instead of calling this function, consider
            /// using a different data structure.
            pub fn countChildren(node: *const Node) usize {
                var count: usize = 0;
                var it: ?*const Node = node.next;
                while (it) |n| : (it = n.next) {
                    count += 1;
                }
                return count;
            }

            /// Reverse the list starting from this node in-place.
            ///
            /// This operation is O(N). Instead of calling this function, consider
            /// using a different data structure.
            pub fn reverse(indirect: *?*Node) void {
                if (indirect.* == null) {
                    return;
                }
                var current: *Node = indirect.*.?;
                while (current.next) |next| {
                    current.next = next.next;
                    next.next = indirect.*;
                    indirect.* = next;
                }
            }
        };

        pub fn prepend(list: *Self, new_node: *Node) void {
            new_node.next = list.first;
            list.first = new_node;
        }

        pub fn remove(list: *Self, node: *Node) void {
            if (list.first == node) {
                list.first = node.next;
            } else {
                var current_elm = list.first.?;
                while (current_elm.next != node) {
                    current_elm = current_elm.next.?;
                }
                current_elm.next = node.next;
            }
        }

        /// Remove and return the first node in the list.
        pub fn popFirst(list: *Self) ?*Node {
            const first = list.first orelse return null;
            list.first = first.next;
            return first;
        }

        /// Iterate over all nodes, returning the count.
        ///
        /// This operation is O(N). Consider tracking the length separately rather than
        /// computing it.
        pub fn len(list: Self) usize {
            if (list.first) |n| {
                return 1 + n.countChildren();
            } else {
                return 0;
            }
        }
    };
}
