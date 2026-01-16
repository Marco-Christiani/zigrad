/// StableHLO IM Verification
///
/// Validates that an MLIR module conforms to the StableHLO IM profile:
/// - Only allowed dialects: func, stablehlo
/// - All custom_call targets are registered (when check is enabled)
/// - No toolchain-specific attributes leaked from PR
///
/// Note: Full operation-level verification requires MLIR bindings to support
/// region/block iteration. Currently, we rely on MLIR's built-in verification
/// and provide a custom_call registry for target validation during lowering.
const std = @import("std");
const mlir = @import("../../ffi/mlir/mlir.zig");

pub const VerifyError = error{
    /// custom_call target is not registered
    UnregisteredCustomCallTarget,
    /// Module failed basic MLIR verification
    InvalidMlir,
    /// Allocation failure
    OutOfMemory,
};

/// Allowed dialect prefixes for the StableHLO IM profile
pub const allowed_dialects = [_][]const u8{
    "func",
    "stablehlo",
    "arith",
    "builtin",
};

/// Registry of allowed custom_call targets.
var custom_call_registry: ?std.StringHashMap(void) = null;

/// Register a custom_call target as allowed.
/// Call this before lowering to register expected custom_call handlers.
pub fn registerCustomCallTarget(target: []const u8) !void {
    if (custom_call_registry == null) {
        custom_call_registry = std.StringHashMap(void).init(std.heap.page_allocator);
    }
    const owned = try std.heap.page_allocator.dupe(u8, target);
    try custom_call_registry.?.put(owned, {});
}

/// Clear all registered custom_call targets
pub fn clearCustomCallTargets() void {
    if (custom_call_registry) |*reg| {
        var it = reg.keyIterator();
        while (it.next()) |key| {
            std.heap.page_allocator.free(key.*);
        }
        reg.deinit();
        custom_call_registry = null;
    }
}

/// Check if a custom_call target is registered
pub fn isCustomCallTargetRegistered(target: []const u8) bool {
    if (custom_call_registry) |reg| {
        return reg.contains(target);
    }
    return false;
}

/// Check if an operation name uses an allowed dialect
pub fn isAllowedDialect(op_name: []const u8) bool {
    for (allowed_dialects) |dialect| {
        if (op_name.len > dialect.len and
            std.mem.startsWith(u8, op_name, dialect) and
            op_name[dialect.len] == '.')
        {
            return true;
        }
    }
    // Built-in operations (no dot separator) are allowed
    if (std.mem.indexOf(u8, op_name, ".") == null) {
        return true;
    }
    return false;
}

/// Verification options
pub const VerifyOptions = struct {
    /// Check that all custom_call targets are registered.
    /// Note: This check happens during lowering, not during module verification.
    check_custom_calls: bool = true,
    /// Run basic MLIR verification
    check_mlir_valid: bool = true,
};

/// Verify that an MLIR module conforms to the StableHLO IM profile.
///
/// This performs basic MLIR verification. For full dialect and custom_call
/// validation, use the registry functions during lowering.
pub fn verifyModule(module: mlir.Module, options: VerifyOptions) VerifyError!void {
    _ = options;
    // Basic MLIR verification
    if (!module.op().verify()) {
        return error.InvalidMlir;
    }
    // Note: Full operation-by-operation verification would require
    // MLIR bindings to support iterating over regions and blocks.
    // For now, we trust MLIR's dialect loading to reject unknown dialects.
}

/// Convenience: verify with default options
pub fn verify(module: mlir.Module) VerifyError!void {
    return verifyModule(module, .{});
}

// Tests =========================================================================

test "allowed dialects" {
    try std.testing.expect(isAllowedDialect("func.func"));
    try std.testing.expect(isAllowedDialect("func.return"));
    try std.testing.expect(isAllowedDialect("stablehlo.add"));
    try std.testing.expect(isAllowedDialect("stablehlo.custom_call"));
    try std.testing.expect(isAllowedDialect("arith.constant"));
    try std.testing.expect(isAllowedDialect("builtin.module"));
    try std.testing.expect(!isAllowedDialect("mhlo.add")); // MHLO not allowed
    try std.testing.expect(!isAllowedDialect("tosa.add")); // TOSA not allowed
}

test "custom call registry" {
    defer clearCustomCallTargets();

    try std.testing.expect(!isCustomCallTargetRegistered("my.target"));
    try registerCustomCallTarget("my.target");
    try std.testing.expect(isCustomCallTargetRegistered("my.target"));
    try std.testing.expect(!isCustomCallTargetRegistered("other.target"));
}
