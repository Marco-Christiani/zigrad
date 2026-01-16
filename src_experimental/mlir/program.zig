/// MLIR/StableHLO Program Loader
///
/// For M1-M3 milestones, this just loads prebuilt MLIR files from disk.
/// Future: In-process MLIR construction via MLIR C-API.

const std = @import("std");

pub const Program = struct {
    format: Format,
    bytecode: []const u8,
    allocator: std.mem.Allocator,

    pub const Format = enum {
        mlir_text,
        mlir_bytecode,
        stablehlo_portable,

        pub fn detectFromExtension(path: []const u8) ?Format {
            if (std.mem.endsWith(u8, path, ".mlir")) return .mlir_text;
            if (std.mem.endsWith(u8, path, ".mlirbc")) return .mlir_bytecode;
            if (std.mem.endsWith(u8, path, ".stablehlo")) return .stablehlo_portable;
            return null;
        }

        pub fn detectFromMagic(data: []const u8) ?Format {
            // MLIR bytecode magic: "ML\xEFR"
            if (data.len >= 4 and std.mem.eql(u8, data[0..4], "ML\xEFR")) {
                return .mlir_bytecode;
            }

            // StableHLO portable artifact.
            // Often wrapped in MLIR bytecode. For now, treat as bytecode
            if (data.len >= 4 and std.mem.eql(u8, data[0..4], "ML\xEFR")) {
                return .stablehlo_portable;
            }

            // Text format. heuristic: starts with "module" or "func"
            if (data.len >= 6) {
                if (std.mem.startsWith(u8, data, "module") or std.mem.startsWith(u8, data, "func.func")) {
                    return .mlir_text;
                }
            }

            return null;
        }
    };

    /// Load program from file
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8) !Program {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const data = try file.readToEndAlloc(allocator, 1 << 30); // 1GB max
        errdefer allocator.free(data);

        // Detect format from extension, fall back to magic bytes
        const format = Format.detectFromExtension(path) orelse
            Format.detectFromMagic(data) orelse
            return error.UnknownFormat;

        return Program{
            .format = format,
            .bytecode = data,
            .allocator = allocator,
        };
    }

    /// Create from in-memory bytecode
    pub fn fromBytecode(allocator: std.mem.Allocator, format: Format, data: []const u8) !Program {
        const bytecode = try allocator.dupe(u8, data);
        return Program{
            .format = format,
            .bytecode = bytecode,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Program) void {
        self.allocator.free(self.bytecode);
    }
};

test "Program.fromBytecode" {
    const allocator = std.testing.allocator;

    const mlir_text =
        \\func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\  return %arg0 : tensor<4xf32>
        \\}
    ;

    var program = try Program.fromBytecode(allocator, .mlir_text, mlir_text);
    defer program.deinit();

    try std.testing.expectEqual(Program.Format.mlir_text, program.format);
    try std.testing.expect(program.bytecode.len > 0);
}
