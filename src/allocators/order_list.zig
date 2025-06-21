//! Copy of std intrusive doubly linked list with API limitations.
//!
//! Used for block_pool orders. Non-circular and used for arbitrary
//! block popping from orders (no need for last data member).

const OrderList = @This();

first: ?*Node = null,

/// This struct contains only the prev and next pointers and not any data
/// payload. The intended usage is to embed it intrusively into another data
/// structure and access the data with `@fieldParentPtr`.
pub const Node = struct {
    prev: ?*Node = null,
    next: ?*Node = null,
};

fn insert_before(list: *OrderList, existing_node: *Node, new_node: *Node) void {
    new_node.next = existing_node;
    if (existing_node.prev) |prev_node| {
        // Intermediate node.
        new_node.prev = prev_node;
        prev_node.next = new_node;
    } else {
        // First element of the list.
        new_node.prev = null;
        list.first = new_node;
    }
    existing_node.prev = new_node;
}

/// Insert a new node at the beginning of the list.
///
/// Arguments:
///     new_node: Pointer to the new node to insert.
pub fn prepend(list: *OrderList, new_node: *Node) void {
    if (list.first) |first| {
        // Insert before first.
        list.insert_before(first, new_node);
    } else {
        // Empty list.
        list.first = new_node;
        new_node.prev = null;
        new_node.next = null;
    }
}

/// Remove a node from the list.
///
/// Arguments:
///     node: Pointer to the node to be removed.
pub fn remove(list: *OrderList, node: *Node) void {
    if (node.prev) |prev_node| {
        // Intermediate node.
        prev_node.next = node.next;
    } else {
        // First element of the list.
        list.first = node.next;
    }

    if (node.next) |next_node| {
        // Intermediate node.
        next_node.prev = node.prev;
    }
}

/// Remove and return the first node in the list.
///
/// Returns:
///     A pointer to the first node in the list.
pub fn pop(list: *OrderList) ?*Node {
    const first = list.first orelse return null;
    list.remove(first);
    return first;
}

/// Iterate over all nodes, returning the count.
///
/// This operation is O(N). Consider tracking the length separately rather than
/// computing it.
pub fn len(list: OrderList) usize {
    var count: usize = 0;
    var it: ?*const Node = list.first;
    while (it) |n| : (it = n.next) count += 1;
    return count;
}
