const std = @import("std");

pub const Op = enum {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    TANH,
};

const grad_clip_delta: f64 = 1000.0;

pub const Value = struct {
    const Self = @This();
    value: f64,
    grad: f64 = 0.0,
    op: ?Op = null,
    children: ?[]*Value = null,
    parent: ?*Value = null,
    label: ?[]const u8 = null,
    _backward: ?*const fn (*Self) void = null,
    alloc: *const std.mem.Allocator,
    incomingEdges: usize = 0,
    detached: bool = false,

    pub fn init(allocator: *const std.mem.Allocator, value: f64, label: ?[]const u8) !*Self {
        const self = try allocator.create(Value);
        self.* = Value{
            .value = value,
            .alloc = allocator,
        };
        _ = self.setLabel(label);
        return self;
    }

    pub fn setLabel(self: *Self, label: ?[]const u8) *Self {
        if (self.label) |l| {
            self.alloc.free(l);
        }
        if (label) |lab| {
            self.label = std.fmt.allocPrint(self.alloc.*, "{s}", .{lab}) catch {
                @panic("Failed to set label");
            };
        } else {
            const uid = generateASCIIUUID(4);
            self.label = std.fmt.allocPrint(self.alloc.*, "{s}", .{uid}) catch {
                @panic("Failed to set label");
            };
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        const values = topologicalSort(self.alloc, self);
        // reverse order
        var i = values.len - 1;
        while (i > 0) : (i -= 1) {
            const curr = values[i];

            if (curr.detached and self != curr) { // allow deinit'ing a detached node/subgraph directly
                continue;
            }
            // std.log.debug("destroying {?s} {}/{}", .{ curr.label, i + 1, values.len });
            if (curr.children) |children| {
                self.alloc.free(children);
            }
            if (curr.label) |l| {
                self.alloc.free(l);
            }
            self.alloc.destroy(curr);
        }
        // std.log.debug("destroying self: {?s}", .{self.label});
        if (self.children) |children| {
            self.alloc.free(children);
        }
        self.alloc.free(values);
        if (self.label) |l| {
            self.alloc.free(l);
        }
        self.alloc.destroy(self);
    }

    pub fn print(self: *const Self) void {
        std.debug.print("Value(label={?s}, value={}, grad={} ", .{ self.label, self.value, self.grad });
        std.debug.print("op={?} ", .{self.op});
        std.debug.print("backward={?}", .{self._backward});
        if (self.children) |children| {
            std.debug.print(" children: ", .{});
            for (children) |elem| {
                std.debug.print("{?s} ", .{elem.label});
            }
        }
        std.debug.print(")\n", .{});
    }

    pub fn print_all(self: *const Self) void {
        std.debug.print("label={?s}\nvalue={}\ngrad={}", .{ self.label, self.value, self.grad });
        std.debug.print("\nop={?}", .{self.op});
        std.debug.print("\nbackward={?}", .{self._backward});
        if (self.children) |children| {
            std.debug.print("\nchildren:\n", .{});
            for (children) |elem| {
                elem.print_all();
            }
        }
        std.debug.print("\n", .{});
    }

    pub fn renderD2(self: Self, writer: anytype) void {
        if (self.children) |children| {
            for (children) |elem| {
                if (!elem.detached) {
                    const symbol = switch (self.op.?) {
                        Op.ADD => "+",
                        Op.SUB => "-",
                        Op.MUL => "x",
                        Op.DIV => "/",
                        Op.POW => "^",
                        Op.TANH => "tanh",
                    };
                    writer.print("{?s}<-{?s}: {s}\n", .{ self.label, elem.label, symbol }) catch @panic("writer print failed");
                }
            }
            for (children) |elem| {
                elem.renderD2(writer);
            }
        } else {
            writer.print("{?s}\n", .{self.label}) catch @panic("wrter print failed");
        }
    }
    pub fn print_arrows(self: *const Self) void {
        self.renderD2(std.io.getStdOut().writer());
    }

    pub fn backward(self: *Self) !void {
        var values = topologicalSort(self.alloc, self);
        if (values[0].grad == 0) {
            values[0].grad = 1;
        } else {
            @panic("Gradient of first value was nonzero, why?");
        }
        for (values) |v| {
            const curr = @constCast(v);
            if (curr._backward) |func| {
                func(curr);
                // clip gradients
                if (curr.children) |children| {
                    children[0].grad = @min(grad_clip_delta, @max(-grad_clip_delta, children[0].grad));
                    children[1].grad = @min(grad_clip_delta, @max(-grad_clip_delta, children[1].grad));
                }
            }
        }
        self.alloc.free(values);
    }

    pub fn zero_grad(self: *Self) void {
        const values = topologicalSort(self.alloc, self);
        defer self.alloc.free(values);
        self.grad = 0;

        // standard order
        for (values) |v| {
            var curr = @constCast(v);
            curr.grad = 0;
        }
    }

    pub fn detach(self: *Self) void {
        // or, remove pointer to self in parent's children... self.parent.?.children
        self.detached = true;
    }
};

const crypto = std.crypto;
const seed = 123;
var prng = std.rand.DefaultPrng.init(seed);

pub fn generateASCIIUUID(comptime length: usize) [length]u8 {
    const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const alphabet_len = alphabet.len;
    var random_bytes: [length]u8 = undefined;
    // crypto.random.bytes(&random_bytes);
    prng.random().bytes(&random_bytes);

    for (random_bytes, 0..) |_, i| {
        random_bytes[i] = alphabet[random_bytes[i] % alphabet_len];
    }
    return random_bytes;
}

pub fn add(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) *Value {
    var children = allocator.alloc(*Value, 2) catch {
        @panic("Failed to allocate children for add()");
    };

    children[0] = v1;
    children[1] = v2;

    var out = Value.init(allocator, v1.value + v2.value, null) catch {
        @panic("Failed to allocate children for add()");
    };

    out.op = Op.ADD;
    out.children = children;
    out._backward = add_backward;
    v1.parent = out;
    v2.parent = out;
    return out;
}

pub fn add_backward(v: *Value) void {
    if (v.children) |_| {
        v.children.?[0].grad += v.grad;
        v.children.?[1].grad += v.grad;
    } else {
        @panic("Called add_backward() but no children.");
    }
}

pub fn sub(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) *Value {
    var children = allocator.alloc(*Value, 2) catch {
        @panic("Failed to allocate children for sub()");
    };

    children[0] = v1;
    children[1] = v2;

    var out = Value.init(allocator, v1.value - v2.value, null) catch {
        @panic("Failed to allocate intermediate for sub()");
    };

    out.op = Op.SUB;
    out.children = children;
    out._backward = sub_backward;
    v1.parent = out;
    v2.parent = out;
    return out;
}

pub fn sub_backward(v: *Value) void {
    if (v.children) |_| {
        v.children.?[0].grad += v.grad;
        v.children.?[1].grad -= v.grad;
    } else {
        @panic("Called sub_backward() but no children.");
    }
}

pub fn mul(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) *Value {
    var children = allocator.alloc(*Value, 2) catch {
        @panic("Failed to allocate children for mul()");
    };
    children[0] = v1;
    children[1] = v2;

    var out = Value.init(allocator, v1.value * v2.value, null) catch {
        @panic("Failed to allocate intermediate for mul()");
    };

    out.op = Op.MUL;
    out.children = children;
    out._backward = mul_backward;
    v1.parent = out;
    v2.parent = out;
    return out;
}

pub fn mul_backward(v: *Value) void {
    if (v.children) |_| {
        v.children.?[0].grad += v.children.?[1].value * v.grad;
        v.children.?[1].grad += v.children.?[0].value * v.grad;
    } else {
        @panic("Called mul_backward() but no children.");
    }
}

pub fn div(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) *Value {
    var children = allocator.alloc(*Value, 2) catch {
        @panic("Failed to allocate children for div()");
    };

    children[0] = v1;
    children[1] = v2;

    var out = Value.init(allocator, v1.value / v2.value, null) catch {
        @panic("Failed to allocate intermediate for div()");
    };

    out.op = Op.DIV;
    out.children = children;
    out._backward = div_backward;
    v1.parent = out;
    v2.parent = out;
    return out;
}

pub fn div_backward(v: *Value) void {
    if (v.children) |_| {
        const a: *Value = v.children.?[0];
        const b: *Value = v.children.?[1];
        v.children.?[0].grad += (1 / b.value) * v.grad;
        v.children.?[1].grad += (-a.value / (b.value * b.value)) * v.grad;
    } else {
        @panic("Called div_backward() but no children.");
    }
}

pub fn pow(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) *Value {
    var children = allocator.alloc(*Value, 2) catch {
        @panic("Failed to allocate children for pow()");
    };

    children[0] = v1;
    children[1] = v2;

    var out = Value.init(allocator, std.math.pow(f64, v1.value, v2.value), null) catch {
        @panic("Failed to allocate intermediate for pow()");
    };

    out.op = Op.POW;
    out.children = children;
    out._backward = pow_backward;
    v1.parent = out;
    v2.parent = out;
    return out;
}

pub fn pow_backward(v: *Value) void {
    // dv/da = b * a^(b-1)
    // dv/db = a^b * log(a) [nb, value = a^b]
    if (v.children) |_| {
        var a: *Value = v.children.?[0];
        var b: *Value = v.children.?[1];
        a.grad += (b.value) * (std.math.pow(f64, a.value, b.value - 1)) * v.grad;
        if (a.value > 0) { // cant take log() of negative
            b.grad += (v.value * @log(a.value)) * v.grad; // these should be identical
            // b.grad += (std.math.pow(f64, a.value, b.value) * @log(a.value)) * v.grad;
        }
    } else {
        @panic("Called pow_backward() but no children.");
    }
}
pub fn tanh(allocator: *const std.mem.Allocator, v: *Value) *Value {
    const x = v.value;
    const val = (std.math.exp(2 * x) - 1) / (std.math.exp(2 * x) + 1);
    var out = Value.init(allocator, val, null) catch {
        @panic("Failed to allocate intermediate for tanh()");
    };

    out.op = Op.TANH;
    out.children = allocator.alloc(*Value, 1) catch {
        @panic("Failed to allocate children for tanh()");
    };

    out.children.?[0] = v;
    v.parent = out;
    return out;
}

pub fn tanh_backward(v: *Value) void {
    if (v.children) |children| {
        v.grad += (1 - std.math.pow(v.value, 2)) * children.?[0].grad;
    } else {
        @panic("Called tanh_backward() but no children.");
    }
}

pub fn dfs(allocator: *const std.mem.Allocator, v: *const Value) ![]*const Value {
    var to_visit_stack = std.ArrayList(*const Value).init(@constCast(allocator).*);
    defer to_visit_stack.deinit();
    var visited = std.AutoArrayHashMap(*const Value, void).init(@constCast(allocator).*); // set
    defer visited.deinit();

    try to_visit_stack.append(v);
    var next: ?*const Value = to_visit_stack.popOrNull();

    while (next != null) : (next = to_visit_stack.popOrNull()) {
        const curr = next.?;
        if (visited.contains(curr)) {
            continue;
        }
        try visited.put(curr, {});
        if (curr.children) |children| {
            var i: usize = children.len;
            while (i > 0) : (i -= 1) {
                if (visited.contains(children[i - 1])) {
                    continue;
                }
                try to_visit_stack.append(children[i - 1]);
            }
        }
    }
    // gotta return copy
    const keys = visited.keys();
    const result = try allocator.alloc(*const Value, keys.len);
    @memcpy(result, keys);
    return result;
}

pub const Neuron = struct {
    w: *Value,
    b: *Value,
};

pub const BatchCallback = fn (value: *const Value, epoch_i: usize) anyerror!void;

pub fn linearModel(allocator: *const std.mem.Allocator, comptime batch_callback: ?BatchCallback) !Neuron {
    const data = @embedFile("data.csv");
    std.debug.print("{s}\n", .{data[0..16]});
    const N = 10;

    const lr: f64 = 1e-2;
    const batchsize = 10;
    const n_epochs = 5;
    var wv = try Value.init(allocator, 0.1, null);
    var bv = try Value.init(allocator, 0.0, null);
    var loss = try Value.init(allocator, 0.0, "l"); // placeholder
    defer loss.deinit(); // deallocate last result
    for (0..n_epochs) |e| {
        var batchy = try allocator.alloc(*Value, batchsize);
        defer allocator.free(batchy);
        var batchyh = try allocator.alloc(*Value, batchsize);
        defer allocator.free(batchyh);

        var data_iter = std.mem.tokenizeScalar(u8, data, '\n');
        var i: usize = 0;

        // loss.deinit(); // Deallocate prev epoch loss

        var steps: usize = 0;
        while (data_iter.next()) |line| : (i += 1) {
            if (steps >= N) break; // HACK: mocks smaller dataset size
            steps += 1;
            var row_iter = std.mem.tokenizeScalar(u8, line, ',');
            const x = try std.fmt.parseFloat(f64, row_iter.next().?);
            const y = try std.fmt.parseFloat(f64, row_iter.next().?);
            const vx = try Value.init(allocator, x, null);
            const vy = try Value.init(allocator, y, null);
            const temp = mul(allocator, vx, wv);
            const pred = add(allocator, temp, bv);

            batchy[i] = vy;
            batchyh[i] = pred;
            if (i >= batchsize - 1) {
                loss = try loss_mse(allocator, batchyh, batchy);
                std.log.warn("loss.value={d:.3}", .{loss.value});
                try loss.backward();
                wv.value -= lr * wv.grad;
                bv.value -= lr * bv.grad;
                std.log.warn("w={d:.3} b={d:.3} wgrad={d:.3} bgrad={d:.3}", .{ wv.value, bv.value, wv.grad, bv.grad });
                // TODO: Lazy, fix epoch callback scope stuff
                if (batch_callback != null) {
                    wv.label = try std.fmt.allocPrint(allocator.*, "w-e{d}", .{e});
                    bv.label = try std.fmt.allocPrint(allocator.*, "b-e{d}", .{e});
                    try batch_callback.?(loss, e);
                    allocator.free(wv.label.?);
                    allocator.free(bv.label.?);
                    wv.label = null;
                    bv.label = null;
                }

                loss.zero_grad();
                i = 0;
                break;
            }
        }
        std.log.warn("({}) w={d:.3} b={d:.3}", .{ e, wv.value, bv.value });
    }
    return Neuron{
        .w = wv,
        .b = bv,
    };
}

pub fn loss_mse(allocator: *const std.mem.Allocator, preds: []*Value, targets: []*Value) !*Value {
    // comp graph deinit should prevent leak later (but this looks like it leaks I know)
    var loss = try Value.init(allocator, 0.0, null);
    for (preds, targets) |yh, y| {
        const diff = sub(allocator, yh, y);
        const sq = pow(allocator, diff, try Value.init(allocator, 2.0, null));
        loss = add(allocator, loss, sq);
    }
    loss = div(allocator, loss, try Value.init(allocator, @floatFromInt(preds.len), null));
    return loss;
}

fn initializeIncomingEdges(root: *Value, allocator: *const std.mem.Allocator) void {
    var to_visit = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer to_visit.deinit();
    var visited = std.AutoArrayHashMap(*Value, bool).init(@constCast(allocator).*);
    defer visited.deinit();

    to_visit.append(root) catch {
        @panic("initializeIncomingEdges() failed to append root node.");
    };
    while (to_visit.items.len > 0) {
        const current = to_visit.pop();
        if (visited.contains(current)) continue;
        visited.put(current, true) catch unreachable;

        if (current.children) |children| {
            for (children) |child| {
                child.incomingEdges += 1;
                to_visit.append(child) catch {
                    @panic("initializeIncomingEdges() failed to append node.");
                };
            }
        }
    }
}

fn topologicalSort(allocator: *const std.mem.Allocator, root: *Value) []*Value {
    initializeIncomingEdges(root, allocator);

    var queue = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer queue.deinit();
    var result = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer result.deinit();

    if (root.incomingEdges == 0) queue.append(root) catch unreachable;

    while (queue.items.len > 0) {
        const current = queue.pop();

        result.append(current) catch unreachable;
        if (current.children) |children| {
            for (children) |child| {
                child.incomingEdges -= 1;
                if (child.incomingEdges == 0) {
                    queue.append(child) catch unreachable;
                }
            }
        }
    }

    return result.toOwnedSlice() catch unreachable;
}

pub fn serializeValueToJson(allocator: std.mem.Allocator, value: *const Value) !std.json.Value {
    var jsonObj = std.json.ObjectMap.init(allocator);
    try jsonObj.put("label", std.json.Value{ .string = value.label orelse "" });
    try jsonObj.put("value", std.json.Value{ .float = value.value });
    try jsonObj.put("grad", std.json.Value{ .float = value.grad });
    try jsonObj.put("op", std.json.Value{ .string = if (value.op) |op| @tagName(op) else "" });

    // Serialize children
    if (value.children) |children| {
        var jsonChildren = std.json.Array.init(allocator);
        errdefer jsonChildren.deinit();
        for (children) |child| {
            const childJson = try serializeValueToJson(allocator, child);
            try jsonChildren.append(childJson);
        }
        try jsonObj.put("children", std.json.Value{ .array = jsonChildren });
    }
    return std.json.Value{ .object = jsonObj };
}

// -----------------------------------------------------------------------------

test "test dfs" {
    const allocator = &std.testing.allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const e = mul(allocator, a, b).setLabel("e");
    const order = try dfs(allocator, e);
    defer allocator.free(order);

    std.debug.print("\n", .{});
    for (order) |item| {
        std.debug.print("{?s}\n", .{item.label});
    }
    e.deinit();
}

test "test single back" {
    const allocator = &std.testing.allocator;
    var a = try Value.init(allocator, 2.0, "a");
    var b = try Value.init(allocator, -3.0, "b");
    var e = mul(allocator, a, b).setLabel("e");
    e.print();
    a.print();
    b.print();
    e._backward.?(e); // design patterns? is that spanish?
    e.print();
    a.print();
    b.print();
    e.deinit();
}

test "test graph" {
    const allocator = &std.testing.allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = try Value.init(allocator, 10, "c");
    const e = mul(allocator, a, b).setLabel("e");
    const d = add(allocator, e, c).setLabel("d");
    const f = try Value.init(allocator, -2.0, "f");
    var L = mul(allocator, d, f).setLabel("L");
    L.print_arrows();
    var json = try serializeValueToJson(allocator.*, L);
    defer json.object.deinit();
    std.debug.print("json:\n", .{});
    try std.json.stringify(json, .{}, std.io.getStdOut().writer());

    try std.testing.expectEqual(e.value, a.value * b.value);
    try std.testing.expectEqual(d.value, (a.value * b.value) + c.value);
    try std.testing.expectEqual(L.value, f.value * ((a.value * b.value) + c.value));

    std.log.info("Attempting deinit\n", .{});
    L.deinit();
    std.log.info("Success\n", .{});
    // This should fail
    // std.log.info("{?s}", .{a.label});
}

test "test mul" {
    const allocator = &std.testing.allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = mul(allocator, a, b).setLabel("c");

    try std.testing.expectEqual(c.value, a.value * b.value);

    std.log.info("Attempting deinit", .{});
    c.deinit();
    std.log.info("Success", .{});
}

test "test add" {
    const allocator = &std.testing.allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = add(allocator, a, b).setLabel("c");

    try std.testing.expectEqual(c.value, a.value + b.value);

    std.log.info("Attempting deinit", .{});
    c.deinit();
    std.log.info("Success", .{});
}

test "test topo" {
    const allocator = &std.testing.allocator;

    // Create nodes
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, 3.0, "b");
    const c = add(allocator, a, b).setLabel("c");
    const d = try Value.init(allocator, 2.0, "d");
    const e = mul(allocator, c, d).setLabel("e");
    // introduce another path
    var f = mul(allocator, d, e).setLabel("f");

    const order = topologicalSort(allocator, f);
    // var order = try dfs(allocator, e);
    defer allocator.free(order);
    for (order, 0..) |value, i| {
        std.debug.print("{}: {?s}\n", .{ i + 1, value.label });
    }

    f.deinit();
}

test "test backprop" {
    const allocator = &std.testing.allocator;
    var w = try Value.init(allocator, 1.0, "w");
    var b = try Value.init(allocator, 1.0, "b");
    var x = try Value.init(allocator, 2.0, "x");
    var y = try Value.init(allocator, -3.0, "y");
    var temp = mul(allocator, x, w).setLabel("t");
    var pred = add(allocator, temp, b).setLabel("p");
    var err = sub(allocator, y, pred).setLabel("e");
    err.print();
    pred.print();
    temp.print();
    try err.backward();
    std.debug.print("Post backward:\n", .{});
    err.print();
    y.print();
    pred.print();
    temp.print();
    b.print();
    x.print();
    w.print();

    try std.testing.expect(1.0 == y.grad);
    try std.testing.expect(-1.0 == pred.grad);
    try std.testing.expect(-1.0 == b.grad);
    try std.testing.expect(-2.0 == w.grad);
    try std.testing.expect(-1.0 == x.grad);
    defer err.deinit();
}

test "test lm" {
    const allocator = &std.testing.allocator;
    _ = try linearModel(allocator, null);
}

// test "test print" {
//     const allocator = &std.testing.allocator;
//     const w = try Value.init(allocator, 1.0, "w");
//     const b = try Value.init(allocator, 1.0, "b");
//     const x = try Value.init(allocator, 2.0, "x");
//     const y = try Value.init(allocator, -3.0, "y");
//     var temp = mul(allocator, x, w).setLabel("t");
//     var pred = add(allocator, temp, b).setLabel("p");
//     var err = sub(allocator, y, pred).setLabel("e");
//     defer err.deinit();
//     try err.backward();

//     // TODO: Free json
//     var graphJson = try serializeValueToJson(allocator.*, err);
//     defer graphJson.object.deinit();
//     const stdout = std.io.getStdOut().writer();
//     try std.json.stringify(graphJson, .{}, stdout);
//     // write to a file
//     // const file = try std.fs.cwd().createFile("output.json", .{});
//     // defer file.close();

//     // const fileWriter = file.writer();
//     // try std.json.stringify(graphJson, .{}, fileWriter);
// }
