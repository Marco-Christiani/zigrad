const std = @import("std");

const Op = enum {
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
    label: ?[]const u8 = null,
    _backward: ?*const fn (*Self) void = null,
    alloc: *const std.mem.Allocator,
    incomingEdges: usize = 0,

    pub fn init(allocator: *const std.mem.Allocator, value: f64, label: ?[]const u8) !*Self {
        const self = try allocator.create(Value);
        self.* = Value{
            .value = value,
            .label = label,
            .alloc = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        const values = topologicalSort(self.alloc, self) catch {
            @panic("Bad things happened");
        };

        // reverse order
        var i = values.len - 1;
        while (i > 0) : (i -= 1) {
            const curr = values[i];
            std.log.debug("destroying {?s} {}/{}", .{ curr.label, i + 1, values.len });
            if (curr.children) |children| {
                self.alloc.free(children);
            }
            self.alloc.destroy(curr);
        }
        std.log.debug("destroying self: {?s}", .{self.label});
        if (self.children) |children| {
            self.alloc.free(children);
        }
        self.alloc.free(values);
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

    pub fn print_arrows(self: *const Self) void {
        // TODO: handle empty label
        if (self.children) |children| {
            for (children) |elem| {
                std.debug.print("{?s}<-{?s}", .{ self.label, elem.label });
                const symbol = switch (self.op.?) {
                    Op.ADD => ": +",
                    Op.SUB => ": -",
                    Op.MUL => ": x",
                    Op.DIV => ": /",
                    Op.POW => ": ^",
                    Op.TANH => ": tanh",
                };
                std.debug.print("{?s}\n", .{symbol});
            }
            for (children) |elem| {
                elem.print_arrows();
            }
        } else {
            std.debug.print("{?s}\n", .{self.label});
        }
    }

    pub fn backward(self: *Self) !void {
        var values = try topologicalSort(self.alloc, self);
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

    pub fn zero_grad(self: *Self) !void {
        const values = try topologicalSort(self.alloc, self);
        defer self.alloc.free(values);
        self.grad = 0;

        // standard order
        for (values) |v| {
            var curr = @constCast(v);
            curr.grad = 0;
        }
    }
};

pub fn add(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) !*Value {
    var children = try allocator.alloc(*Value, 2);
    children[0] = v1;
    children[1] = v2;

    var out = try Value.init(allocator, v1.value + v2.value, null);
    out.op = Op.ADD;
    out.children = children;
    out._backward = add_backward;
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

pub fn sub(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) !*Value {
    var children = try allocator.alloc(*Value, 2);
    children[0] = v1;
    children[1] = v2;

    var out = try Value.init(allocator, v1.value - v2.value, null);
    out.op = Op.SUB;
    out.children = children;
    out._backward = sub_backward;
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

pub fn mul(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) !*Value {
    var children = try allocator.alloc(*Value, 2);
    children[0] = v1;
    children[1] = v2;

    var out = try Value.init(allocator, v1.value * v2.value, null);
    out.op = Op.MUL;
    out.children = children;
    out._backward = mul_backward;
    return out;
}

pub fn mul_backward(v: *Value) void {
    std.log.info("mul_backward(): Entered", .{});
    if (v.children) |_| {
        std.log.info("mul_backward(): Got children", .{});
        v.children.?[0].grad += v.children.?[1].value * v.grad;
        v.children.?[1].grad += v.children.?[0].value * v.grad;
    } else {
        @panic("Called mul_backward() but no children.");
    }
}

pub fn div(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) !*Value {
    var children = try allocator.alloc(*Value, 2);
    children[0] = v1;
    children[1] = v2;

    var out = try Value.init(allocator, v1.value / v2.value, null);
    out.op = Op.DIV;
    out.children = children;
    out._backward = div_backward;
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

pub fn pow(allocator: *const std.mem.Allocator, v1: *Value, v2: *Value) !*Value {
    var children = try allocator.alloc(*Value, 2);
    children[0] = v1;
    children[1] = v2;

    var out = try Value.init(allocator, std.math.pow(f64, v1.value, v2.value), null);
    out.op = Op.POW;
    out.children = children;
    out._backward = pow_backward;
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
pub fn tanh(allocator: *const std.mem.Allocator, v: *Value) !*Value {
    const x = v.value;
    const val = (std.math.exp(2 * x) - 1) / (std.math.exp(2 * x) + 1);
    var out = try Value.init(allocator, val, null);
    out.op = Op.TANH;
    out.children = try allocator.alloc(*Value, 1);
    out.children.?[0] = v;
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

pub const EpochCallback = fn (value: *const Value, epoch_i: usize) anyerror!void;

pub fn linearModel(allocator: *const std.mem.Allocator, comptime epoch_callback: ?EpochCallback) !Neuron {
    const data = @embedFile("data.csv");
    std.debug.print("{s}\n", .{data[0..16]});

    const lr: f64 = 1e-2;
    const batchsize = 50;
    const n_epochs = 1;
    var wv = try Value.init(allocator, 0.1, "w");
    var bv = try Value.init(allocator, 0.0, "b");
    var loss = try Value.init(allocator, 0.0, "l"); // placeholder
    defer loss.deinit(); // deallocate last result
    for (0..n_epochs) |e| {
        var batchy = try allocator.alloc(*Value, batchsize);
        defer allocator.free(batchy);
        var batchyh = try allocator.alloc(*Value, batchsize);
        defer allocator.free(batchyh);

        var data_iter = std.mem.tokenizeScalar(u8, data, '\n');
        var i: usize = 0;

        loss.deinit(); // Deallocate prev epoch loss

        while (data_iter.next()) |line| : (i += 1) {
            var row_iter = std.mem.tokenizeScalar(u8, line, ',');
            const x = try std.fmt.parseFloat(f64, row_iter.next().?);
            const y = try std.fmt.parseFloat(f64, row_iter.next().?);
            const vx = try Value.init(allocator, x, null);
            const vy = try Value.init(allocator, y, null);
            const temp = try mul(allocator, vx, wv);
            const pred = try add(allocator, temp, bv);

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
                if (epoch_callback != null) {
                    try epoch_callback.?(loss, e);
                }

                try loss.zero_grad();
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
        const diff = try sub(allocator, yh, y);
        const sq = try pow(allocator, diff, try Value.init(allocator, 2.0, null));
        loss = try add(allocator, loss, sq);
    }
    loss = try div(allocator, loss, try Value.init(allocator, @floatFromInt(preds.len), null));
    return loss;
}

fn initializeIncomingEdges(root: *Value, allocator: *const std.mem.Allocator) !void {
    var to_visit = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer to_visit.deinit();
    var visited = std.AutoArrayHashMap(*Value, bool).init(@constCast(allocator).*);
    defer visited.deinit();

    try to_visit.append(root);
    while (to_visit.items.len > 0) {
        const current = to_visit.pop();
        if (visited.contains(current)) continue;
        try visited.put(current, true);

        if (current.children) |children| {
            for (children) |child| {
                child.incomingEdges += 1;
                try to_visit.append(child);
            }
        }
    }
}

fn topologicalSort(allocator: *const std.mem.Allocator, root: *Value) ![]*Value {
    try initializeIncomingEdges(root, allocator);

    var queue = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer queue.deinit();
    var result = std.ArrayList(*Value).init(@constCast(allocator).*);
    defer result.deinit();

    if (root.incomingEdges == 0) try queue.append(root);

    while (queue.items.len > 0) {
        const current = queue.pop();

        try result.append(current);
        if (current.children) |children| {
            for (children) |child| {
                child.incomingEdges -= 1;
                if (child.incomingEdges == 0) {
                    try queue.append(child);
                }
            }
        }
    }

    return result.toOwnedSlice();
}

pub fn serializeValueToJson(allocator: std.mem.Allocator, value: *const Value) !std.json.Value {
    var jsonObj = std.json.ObjectMap.init(allocator);
    try jsonObj.put("label", std.json.Value{ .string = value.label orelse "" });
    try jsonObj.put("value", std.json.Value{ .float = value.value });
    try jsonObj.put("grad", std.json.Value{ .float = value.grad });

    // Serialize children
    if (value.children) |children| {
        var jsonChildren = std.json.Array.init(allocator);
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
    const allocator = &std.heap.page_allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const e = try mul(allocator, a, b);
    e.label = "e";
    const order = try dfs(allocator, e);

    std.debug.print("\n", .{});
    for (order) |item| {
        std.debug.print("{?s}\n", .{item.label});
    }
}

test "test single back" {
    const allocator = &std.heap.page_allocator;
    var a = try Value.init(allocator, 2.0, "a");
    var b = try Value.init(allocator, -3.0, "b");
    var e = try mul(allocator, a, b);
    e.label = "e";
    e.print();
    a.print();
    b.print();
    e._backward.?(e); // design patterns? is that spanish?
    e.print();
    a.print();
    b.print();
}

test "test graph" {
    const allocator = &std.testing.allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = try Value.init(allocator, 10, "c");
    const e = try mul(allocator, a, b);
    e.label = "e";
    var d = try add(allocator, e, c);
    d.label = "d";
    const f = try Value.init(allocator, -2.0, "f");
    var L = try mul(allocator, d, f);
    L.label = "L";
    L.print();

    try std.testing.expectEqual(e.value, a.value * b.value);
    try std.testing.expectEqual(d.value, (a.value * b.value) + c.value);
    try std.testing.expectEqual(L.value, f.value * ((a.value * b.value) + c.value));

    std.log.info("Attempting deinit", .{});
    L.deinit();
    std.log.info("Success", .{});
    // This should fail
    // std.log.info("{?s}", .{a.label});
}

test "test mul" {
    const allocator = &std.heap.page_allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = try mul(allocator, a, b);
    c.label = "c";
    c.print();

    try std.testing.expectEqual(c.value, a.value * b.value);

    std.log.info("Attempting deinit", .{});
    c.deinit();
    std.log.info("Success", .{});
}

test "test add" {
    const allocator = &std.heap.page_allocator;
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, -3.0, "b");
    const c = try add(allocator, a, b);
    c.label = "c";
    c.print();

    try std.testing.expectEqual(c.value, a.value + b.value);

    std.log.info("Attempting deinit", .{});
    c.deinit();
    std.log.info("Success", .{});
}

test "test topo" {
    var allocator = &std.heap.page_allocator;

    // Create nodes
    const a = try Value.init(allocator, 2.0, "a");
    const b = try Value.init(allocator, 3.0, "b");
    var c = try add(allocator, a, b);
    c.label = "c";
    var d = try Value.init(allocator, 2.0, "d");
    var e = try mul(allocator, c, d);
    e.label = "e";
    // introduce another path
    var f = try mul(allocator, d, e);
    f.label = "f";

    const order = try topologicalSort(allocator, f);
    // var order = try dfs(allocator, e);
    defer allocator.free(order);
    for (order, 0..) |value, i| {
        std.debug.print("{}: {?s}\n", .{ i + 1, value.label });
    }

    d.deinit();
}

test "test backprop" {
    const allocator = &std.testing.allocator;
    var w = try Value.init(allocator, 1.0, "w");
    var b = try Value.init(allocator, 1.0, "b");
    var x = try Value.init(allocator, 2.0, "x");
    var y = try Value.init(allocator, -3.0, "y");
    var temp = try mul(allocator, x, w);
    temp.label = "t";
    var pred = try add(allocator, temp, b);
    pred.label = "p";
    var err = try sub(allocator, y, pred);
    err.label = "e";
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

test "test print" {
    const allocator = &std.heap.page_allocator;
    const w = try Value.init(allocator, 1.0, "w");
    const b = try Value.init(allocator, 1.0, "b");
    const x = try Value.init(allocator, 2.0, "x");
    const y = try Value.init(allocator, -3.0, "y");
    var temp = try mul(allocator, x, w);
    temp.label = "t";
    var pred = try add(allocator, temp, b);
    pred.label = "p";
    var err = try sub(allocator, y, pred);
    err.label = "err";
    try err.backward();

    // TODO: Free json
    const graphJson = try serializeValueToJson(allocator.*, err);
    const stdout = std.io.getStdOut().writer();
    try std.json.stringify(graphJson, .{}, stdout);
    // write to a file
    const file = try std.fs.cwd().createFile("output.json", .{});
    defer file.close();

    const fileWriter = file.writer();
    try std.json.stringify(graphJson, .{}, fileWriter);
    defer err.deinit();
}
