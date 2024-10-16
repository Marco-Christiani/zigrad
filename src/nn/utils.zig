const std = @import("std");
const zg = @import("../zigrad.zig");

const random = zg.random;
const Op = zg.Op;
const NDTensor = zg.NDTensor;

const log = std.log.scoped(.zigrad_trainer);

pub const PrintOptions = struct {
    add_symbol: []const u8,
    sub_symbol: []const u8,
    mul_symbol: []const u8,
    div_symbol: []const u8,
    sum_symbol: []const u8,
    matmul_symbol: []const u8,
    matvec_symbol: []const u8,
    max_symbol: []const u8 = "max",
    exp_symbol: []const u8 = "exp",
    reshape_symbol: []const u8 = "reshape",
    arrow_symbol: []const u8 = "<-",

    pub const plain = PrintOptions{
        .add_symbol = "+",
        .sub_symbol = "-",
        .mul_symbol = "×",
        .div_symbol = "/",
        .sum_symbol = "++",
        .matmul_symbol = "AB",
        .matvec_symbol = "Ax",
    };
    pub const unicode1 = PrintOptions{
        .add_symbol = "+",
        .sub_symbol = "-",
        .mul_symbol = "×",
        .div_symbol = "÷",
        .sum_symbol = "∑",
        .matmul_symbol = "⊗",
        .matvec_symbol = "⊙",
    };
    const unicode2 = PrintOptions{
        .add_symbol = "➕",
        .sub_symbol = "➖",
        .mul_symbol = "✖️",
        .div_symbol = "➗",
        .sum_symbol = "∑",
        .matmul_symbol = "⊗",
        .matvec_symbol = "⊙",
    };
};

pub fn generateASCIIUUID(comptime length: usize) [length]u8 {
    const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
    const alphabet_len = alphabet.len;
    var random_bytes: [length]u8 = undefined;
    random.bytes(&random_bytes);

    for (random_bytes, 0..) |_, i| {
        random_bytes[i] = alphabet[random_bytes[i] % alphabet_len];
    }
    return random_bytes;
}

fn randomSuffix(label: ?[]const u8, allocator: std.mem.Allocator) []const u8 {
    if (label) |l| {
        if (std.mem.containsAtLeast(u8, l, 1, "__")) return l; // already been labeled
        const uid = generateASCIIUUID(4);
        return std.fmt.allocPrint(allocator, "{s}__{s}", .{ l, uid }) catch {
            @panic("Failed to set label");
        };
    }
    const uid = generateASCIIUUID(4);
    return std.fmt.allocPrint(allocator, "__{s}", .{uid}) catch {
        @panic("Failed to set label");
    };
}

const LabelGenerator = struct {
    map: std.AutoHashMap(*const anyopaque, []const u8),
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) LabelGenerator {
        return .{
            .map = std.AutoHashMap(*const anyopaque, []const u8).init(allocator),
            .allocator = allocator,
        };
    }

    fn deinit(self: *LabelGenerator) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.map.deinit();
    }

    fn getOrCreateLabel(self: *LabelGenerator, ptr: *const anyopaque, base_label: ?[]const u8) ![]const u8 {
        if (self.map.get(ptr)) |label| {
            return label;
        }

        const new_label = randomSuffix(base_label, self.allocator);
        try self.map.put(ptr, new_label);
        return new_label;
    }
};

pub fn printD2(
    node: anytype,
    opts: PrintOptions,
    writer: anytype,
    label_gen: *LabelGenerator,
    visited: *std.AutoHashMap(*const anyopaque, void),
) !void {
    const T = @TypeOf(node);
    const info = @typeInfo(T);
    const S = info.Pointer.child;

    if (!@hasField(S, "label") or !@hasField(S, "op") or !@hasField(S, "children")) {
        @compileError("Struct must have 'label', 'op', and 'children' fields");
    }

    const node_label = try label_gen.getOrCreateLabel(node, node.label);

    if (visited.contains(node)) {
        return; // Skip if we've already visited this node
    }
    try visited.put(node, {});

    if (node.children) |children| {
        for (children) |elem| {
            const elem_label = try label_gen.getOrCreateLabel(elem, elem.label);
            try writer.print("{s}{s}{s}", .{ node_label, opts.arrow_symbol, elem_label });
            if (node.op) |op| {
                const symbol = switch (op) {
                    .ADD => opts.add_symbol,
                    .SUB => opts.sub_symbol,
                    .MUL => opts.mul_symbol,
                    .DIV => opts.div_symbol,
                    .SUM => opts.sum_symbol,
                    .MATMUL_AB, .MATMUL_AtB, .MATMUL_ABt, .MATMUL_AtBt => opts.matmul_symbol,
                    .MATVEC => opts.matvec_symbol,
                    .MAX => opts.max_symbol,
                    .EXP => opts.exp_symbol,
                    .RESHAPE => opts.reshape_symbol,
                    else => "?",
                    // else => if (node._backward) |bwfn| @typeInfo(bwfn),
                };
                try writer.print(": {s}\n", .{symbol});
            } else {
                try writer.print("\n", .{});
            }
        }
        for (children) |elem| {
            try printD2(elem, opts, writer, label_gen, visited);
        }
    } else {
        try writer.print("{s}\n", .{node_label});
    }
}

pub fn renderD2(
    node: anytype,
    opts: PrintOptions,
    allocator: std.mem.Allocator,
    output_file: []const u8,
) !void {
    var d2_code = std.ArrayList(u8).init(allocator);
    defer d2_code.deinit();

    var label_gen = LabelGenerator.init(allocator);
    defer label_gen.deinit();

    var visited = std.AutoHashMap(*const anyopaque, void).init(allocator);
    defer visited.deinit();

    try d2_code.writer().writeAll(
        \\vars: {
        \\  d2-config: {
        \\    theme-id: 101
        \\    dark-theme-id: 101
        \\    pad: 0
        \\    center: true
        \\    sketch: false
        \\    layout-engine: dagre
        \\  }
        \\}
        \\direction: right
        \\style: {
        \\  font-size: 40
        \\  fill: transparent
        \\}
        \\*.style: {
        \\  stroke-width: 4
        \\  font-size: 40
        \\  stroke: "#F7A41D"
        \\}
        \\(* <- *)[*].style: {
        \\  stroke: "#F7A41D"
        \\  font-size: 35
        \\  stroke-width: 5
        \\}
        \\
    );
    log.debug("traversing", .{});
    try printD2(node, opts, d2_code.writer(), &label_gen, &visited);
    log.debug("rendering", .{});
    const d2filepath = try std.fmt.allocPrint(allocator, "{s}/{s}{s}", .{ std.fs.path.dirname(output_file) orelse "", std.fs.path.stem(output_file), ".d2" });
    defer allocator.free(d2filepath);
    log.debug("d2filepath: {s}", .{d2filepath});
    const d2file = try std.fs.cwd().createFile(d2filepath, .{});
    defer d2file.close();

    try d2file.writeAll(d2_code.items);
    const args = [_][]const u8{
        "d2",
        "--theme",
        "101",
        d2filepath,
        output_file,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &args,
    });

    if (result.term.Exited != 0) {
        std.debug.print("d2 CLI failed with exit code: {}\n", .{result.term.Exited});
        std.debug.print("stderr: {s}\n", .{result.stderr});
    } else {
        std.debug.print("Diagram generated: {s}\n", .{output_file});
        allocator.free(result.stdout);
        allocator.free(result.stderr);
    }
}

pub fn sesame(filepath: []const u8, allocator: std.mem.Allocator) !void {
    const opencmd = switch (@import("builtin").os.tag) {
        .linux => "xdg-open",
        .macos => "open",
        else => std.log.err("Unsupported os {}", .{std.Target.Os.Tag}),
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ opencmd, filepath },
    });

    if (result.term.Exited != 0) {
        std.log.err("open failed with exit code: {}\n", .{result.term.Exited});
        std.log.err("stderr: {s}\n", .{result.stderr});
    }

    allocator.free(result.stdout);
    allocator.free(result.stderr);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.log.err("Leak detected.", .{});
        }
    }
    const alloc = gpa.allocator();

    const Node = struct {
        label: ?[]const u8,
        op: ?Op,
        children: ?[2]*const @This(),
    };
    const child1 = Node{ .label = "child1", .op = .MUL, .children = null };
    const child2 = Node{ .label = "child2", .op = .DIV, .children = null };
    var root = Node{
        .label = try alloc.dupe(u8, "root"),
        .op = .ADD,
        .children = [2]*const Node{ &child1, &child2 },
    };
    defer alloc.free(root.label.?);
    try renderD2(&root, PrintOptions.unicode1, alloc, "/tmp/generated1.png");

    const shape = &[_]usize{ 2, 3 };
    const Tensor = NDTensor(f32);

    var t1 = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), shape, false, alloc);
    _ = t1.setLabel("t1");
    defer t1.deinit();
    var t2 = try Tensor.init(@constCast(&[_]f32{ 10, 20, 30, 40, 50, 60 }), shape, false, alloc);
    _ = t2.setLabel("t2");
    defer t2.deinit();

    var t3 = try t1.add(t2, alloc);
    _ = t3.setLabel("t3");
    defer t3.deinit();

    try renderD2(t3, PrintOptions.plain, alloc, "/tmp/generated2.png");
    std.log.info("Done.\n", .{});

    // try sesame("/tmp/generated1.png", alloc);
    // try sesame("/tmp/generated2.png", alloc);
}
