const std = @import("std");
const tensor = @import("tensor.zig");

const Op = tensor.Op;
const NDTensor = tensor.NDTensor;

pub const PrintOptions = struct {
    add_symbol: []const u8,
    sub_symbol: []const u8,
    mul_symbol: []const u8,
    div_symbol: []const u8,
    sum_symbol: []const u8,
    matmul_symbol: []const u8,
    matvec_symbol: []const u8,
    arrow_symbol: []const u8 = "<-",
    pub const plain = PrintOptions{
        .add_symbol = "+",
        .sub_symbol = "-",
        .mul_symbol = "×",
        .div_symbol = "/",
        .sum_symbol = "++",
        .matmul_symbol = "@",
        .matvec_symbol = "@@",
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

pub fn printD2(
    node: anytype,
    opts: PrintOptions,
    writer: anytype,
) !void {
    const T = @TypeOf(node);
    const info = @typeInfo(T);
    // Ensure we're working with a pointer to a struct
    if (info != .Pointer or info.Pointer.size != .One or @typeInfo(info.Pointer.child) != .Struct) {
        @compileError("print_arrows expects a pointer to a struct");
    }
    const S = info.Pointer.child;
    // Check for required fields
    if (!@hasField(S, "label") or !@hasField(S, "op") or !@hasField(S, "children")) {
        @compileError("Struct must have 'label', 'op', and 'children' fields");
    }
    if (node.children) |children| {
        for (children) |elem| {
            try writer.print("{?s}{s}{?s}", .{ node.label, opts.arrow_symbol, elem.label });
            if (node.op) |op| {
                const symbol = switch (op) {
                    .ADD => opts.add_symbol,
                    .SUB => opts.sub_symbol,
                    .MUL => opts.mul_symbol,
                    .DIV => opts.div_symbol,
                    .SUM => opts.sum_symbol,
                    .MATMUL => opts.matmul_symbol,
                    .MATVEC => opts.matvec_symbol,
                    else => "?",
                };
                try writer.print(": {s}\n", .{symbol});
            } else {
                try writer.print("\n", .{});
            }
        }
        for (children) |elem| {
            try printD2(elem, opts, writer);
        }
    } else {
        try writer.print("{?s}\n", .{node.label});
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

    try printD2(node, opts, d2_code.writer());

    const d2filepath = try std.fmt.allocPrint(allocator, "{s}{s}", .{ std.fs.path.stem(output_file), ".d2" });
    defer allocator.free(d2filepath);
    const d2file = try std.fs.cwd().createFile(d2filepath, .{});
    defer d2file.close();

    try d2file.writeAll(d2_code.items);
    const args = [_][]const u8{
        "d2",
        "--theme",
        "200",
        d2filepath,
        output_file,
    };

    const result = try std.ChildProcess.run(.{
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

    const result = try std.ChildProcess.run(.{
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
        .label = "root",
        .op = .ADD,
        .children = [2]*const Node{ &child1, &child2 },
    };
    try renderD2(&root, PrintOptions.unicode1, alloc, "/tmp/generated1.png");

    const shape = &[_]usize{ 2, 3 };
    const Tensor = NDTensor(f32);

    var t1 = try Tensor.init(@constCast(&[_]f32{ 1, 2, 3, 4, 5, 6 }), shape, false, alloc);
    t1.label = "t1";
    defer t1.deinit(alloc);
    var t2 = try Tensor.init(@constCast(&[_]f32{ 10, 20, 30, 40, 50, 60 }), shape, false, alloc);
    t2.label = "t2";
    defer t2.deinit(alloc);

    var t3 = try t1.add(t2, alloc);
    t3.label = "t3";
    defer t3.deinit(alloc);

    try renderD2(t3, PrintOptions.plain, alloc, "/tmp/generated2.png");
    std.log.info("Done.\n", .{});

    // try sesame("/tmp/generated1.png", alloc);
    // try sesame("/tmp/generated2.png", alloc);
    // 2>&1 | rg --color always 'error(gpa)*' 2>&1 | tee /dev/tty | rg --color always --count-matches 'error(gpa)*' 2>&1 | awk '{s+=$1} END {print "Total Matches:", s}'
    // 2>&1 | rg --passthru --color always 'error(gpa)*' 2>&1 | tee /dev/tty | rg --passthru --color always --count-matches 'error(gpa)*' 2>&1 | awk '{s+=$1} END {print "Total Matches:", s}'
}
