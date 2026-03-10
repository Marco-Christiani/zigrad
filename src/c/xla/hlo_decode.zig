/// HLO Protobuf Decoder
///
/// Decodes XLA HLO protobuf bytes into human-readable text.
/// Used by `dump_optimized_program` (pipeline) and the standalone `decode_hlo` tool.
const std = @import("std");
const protobuf = @import("protobuf");
const xla = @import("xla_pb");

pub const HloModuleProto = xla.HloModuleProto;
pub const HloModuleProtoWithConfig = xla.HloModuleProtoWithConfig;

/// Decode XLA HLO protobuf bytes and write human-readable text to `out`.
///
/// Tries `HloModuleProtoWithConfig` first (format from `--dump-optimized`),
/// falls back to plain `HloModuleProto`. Returns `true` on success, `false`
/// if the bytes could not be decoded (caller should fall back to raw output).
pub fn decode_and_print(bytes: []const u8, allocator: std.mem.Allocator, out: *std.Io.Writer) bool {
    // Try HloModuleProtoWithConfig first.
    blk: {
        var reader: std.Io.Reader = .fixed(bytes);
        const with_config = HloModuleProtoWithConfig.decode(&reader, allocator) catch break :blk;
        defer @constCast(&with_config).deinit(allocator);

        if (with_config.hlo_module) |*module| {
            print_module(module, out) catch return false;
            out.flush() catch return false;
            return true;
        }
    }

    // Fall back to plain HloModuleProto.
    var reader: std.Io.Reader = .fixed(bytes);
    const module = HloModuleProto.decode(&reader, allocator) catch return false;
    defer @constCast(&module).deinit(allocator);

    print_module(&module, out) catch return false;
    out.flush() catch return false;
    return true;
}

pub fn print_module(module: *const xla.HloModuleProto, out: *std.Io.Writer) !void {
    try out.print("HloModule \"{s}\", id={d}\n", .{ module.name, module.id });
    if (module.entry_computation_name.len > 0) {
        try out.print("  entry: \"{s}\" (id={d})\n", .{
            module.entry_computation_name,
            module.entry_computation_id,
        });
    }
    try out.print("  computations: {d}\n", .{module.computations.items.len});
    try out.print("  is_dynamic: {}\n\n", .{module.is_dynamic});

    for (module.computations.items) |*comp| {
        try print_computation(comp, out);
    }
}

pub fn print_computation(comp: *const xla.HloComputationProto, out: *std.Io.Writer) !void {
    const tag = if (comp.is_fusion_computation) " [fusion]" else "";
    try out.print("computation \"{s}\" (id={d}, root_id={d}){s}\n", .{
        comp.name,
        comp.id,
        comp.root_id,
        tag,
    });

    if (comp.program_shape) |*ps| {
        try out.print("  signature: (", .{});
        for (ps.parameters.items, 0..) |*p, i| {
            if (i > 0) try out.print(", ", .{});
            try print_shape(p, out);
        }
        try out.print(") -> ", .{});
        if (ps.result) |*r| {
            try print_shape(r, out);
        }
        try out.print("\n", .{});
    }

    for (comp.instructions.items) |*instr| {
        try print_instruction(instr, out);
    }
    try out.print("\n", .{});
}

pub fn print_instruction(instr: *const xla.HloInstructionProto, out: *std.Io.Writer) !void {
    try out.print("  %{s} = {s}", .{ instr.name, instr.opcode });

    // Print operand references
    if (instr.operand_ids.items.len > 0) {
        try out.print("(", .{});
        for (instr.operand_ids.items, 0..) |id, i| {
            if (i > 0) try out.print(", ", .{});
            try out.print("%{d}", .{id});
        }
        try out.print(")", .{});
    }

    // Print shape
    if (instr.shape) |*s| {
        try out.print(" : ", .{});
        try print_shape(s, out);
    }

    // Print extra info for specific opcodes
    if (instr.parameter_number != 0 or std.mem.eql(u8, instr.opcode, "parameter")) {
        try out.print(" param={d}", .{instr.parameter_number});
    }
    if (instr.dimensions.items.len > 0) {
        try out.print(" dims=[", .{});
        for (instr.dimensions.items, 0..) |d, i| {
            if (i > 0) try out.print(",", .{});
            try out.print("{d}", .{d});
        }
        try out.print("]", .{});
    }

    try out.print("\n", .{});
}

pub fn print_shape(shape: *const xla.ShapeProto, out: *std.Io.Writer) !void {
    // tuple_shapes stubbed to raw bytes -- just indicate tuple
    if (shape.tuple_shapes_raw.len > 0) {
        try out.print("tuple(...)", .{});
        return;
    }

    try out.print("{s}", .{@tagName(shape.element_type)});
    if (shape.dimensions.items.len > 0) {
        try out.print("[", .{});
        for (shape.dimensions.items, 0..) |d, i| {
            if (i > 0) try out.print(",", .{});
            try out.print("{d}", .{d});
        }
        try out.print("]", .{});
    }
}
