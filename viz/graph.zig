const std = @import("std");

const MAX_NODES = 1024;
const MAX_EDGES = 4096;
const MAX_NAME_STORAGE = 65536; // 64KB for all node names

var node_count_: u32 = 0;
var edge_count_: u32 = 0;

// Node storage
var node_names: [MAX_NODES][*]const u8 = undefined;
var node_name_lens: [MAX_NODES]u32 = undefined;

// Edge storage
var edge_sources: [MAX_EDGES]u32 = undefined;
var edge_targets: [MAX_EDGES]u32 = undefined;

// Name string pool
var name_pool: [MAX_NAME_STORAGE]u8 = undefined;
var name_pool_ptr: u32 = 0;

// Add a node with a name (copied into name pool)
// Returns the node index
export fn add_node(name_ptr: [*]const u8, name_len: u32) u32 {
    if (node_count_ >= MAX_NODES) return 0xFFFFFFFF;
    if (name_pool_ptr + name_len > MAX_NAME_STORAGE) return 0xFFFFFFFF;

    // Copy name into pool
    var i: u32 = 0;
    while (i < name_len) : (i += 1) {
        name_pool[name_pool_ptr + i] = name_ptr[i];
    }

    // Store reference
    node_names[node_count_] = @ptrCast(&name_pool[name_pool_ptr]);
    node_name_lens[node_count_] = name_len;

    name_pool_ptr += name_len;
    node_count_ += 1;

    return node_count_ - 1;
}

// Add an edge from source to target
// Returns 1 on success, 0 on failure
export fn add_edge(source: u32, target: u32) u32 {
    if (edge_count_ >= MAX_EDGES) return 0;
    if (source >= node_count_ or target >= node_count_) return 0;

    edge_sources[edge_count_] = source;
    edge_targets[edge_count_] = target;
    edge_count_ += 1;

    return 1;
}

// Reset the graph
export fn reset() void {
    node_count_ = 0;
    edge_count_ = 0;
    name_pool_ptr = 0;
}

// Transformer graph definition (in Zig data, not built yet)
const TransformerDef = struct {
    nodes: []const []const u8 = &.{
        "input",
        "token_emb",
        "pos_emb",
        "embed_add",
        "ln1",
        "q_proj",
        "k_proj",
        "v_proj",
        "attention",
    },
    edges: []const [2]u32 = &.{
        .{ 0, 1 },
        .{ 1, 3 },
        .{ 2, 3 },
        .{ 3, 4 },
        .{ 4, 5 },
        .{ 4, 6 },
        .{ 4, 7 },
        .{ 5, 8 },
        .{ 6, 8 },
        .{ 7, 8 },
    },
};

var build_step_: u32 = 0;

// Get the next step to build (returns step number, 0xFFFFFFFF when done)
export fn build_step() u32 {
    const def = TransformerDef{};
    const total_steps = def.nodes.len + def.edges.len;

    if (build_step_ >= total_steps) {
        return 0xFFFFFFFF;
    }

    if (build_step_ < def.nodes.len) {
        // Add a node
        const name = def.nodes[build_step_];
        var i: u32 = 0;
        while (i < name.len) : (i += 1) {
            name_pool[name_pool_ptr + i] = name[i];
        }
        node_names[node_count_] = @ptrCast(&name_pool[name_pool_ptr]);
        node_name_lens[node_count_] = @intCast(name.len);
        name_pool_ptr += @intCast(name.len);
        node_count_ += 1;
    } else {
        // Add an edge
        const edge_idx = build_step_ - def.nodes.len;
        const edge = def.edges[edge_idx];
        edge_sources[edge_count_] = edge[0];
        edge_targets[edge_count_] = edge[1];
        edge_count_ += 1;
    }

    build_step_ += 1;
    return build_step_ - 1;
}

// Build instantly (all at once)
export fn build_transformer() void {
    reset();
    build_step_ = 0;
    while (build_step() != 0xFFFFFFFF) {}
}

// Query functions
export fn node_count() u32 {
    return node_count_;
}

export fn edge_count() u32 {
    return edge_count_;
}

export fn node_name_ptr(i: u32) [*]const u8 {
    if (i >= node_count_) return @as([*]const u8, undefined);
    return node_names[i];
}

export fn node_name_len(i: u32) u32 {
    if (i >= node_count_) return 0;
    return node_name_lens[i];
}

export fn edge_source(i: u32) u32 {
    if (i >= edge_count_) return 0xFFFFFFFF;
    return edge_sources[i];
}

export fn edge_target(i: u32) u32 {
    if (i >= edge_count_) return 0xFFFFFFFF;
    return edge_targets[i];
}
