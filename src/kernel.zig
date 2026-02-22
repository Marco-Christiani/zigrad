/// Kernel Provider Interface
///
/// Defines the extension point for kernel providers — components that can
/// claim PR regions and produce compiled kernel artifacts (KAs) for specific
/// targets.
///
/// This module is independent of the pipeline and backend. It depends only
/// on PR types. The pipeline consumes it (via KernelizePass in
/// pipeline/kernelize.zig), the backend consumes it (to resolve custom_call
/// references), and provider implementations consume it (to implement the
/// KernelProvider interface).
const std = @import("std");
const pr = @import("pr/pr.zig");

// ============================================================================
// Region Descriptor
// ============================================================================

/// A view into a PR subgraph representing a kernelizable region.
///
/// Contains the equations, their types, and the region's boundary
/// (which vars flow in from outside, which flow out). This is the
/// information a kernel provider needs to decide whether it can handle
/// a region and to compile a kernel for it.
///
/// RegionDescriptor does not own any memory — all slices are views into
/// the parent Function's storage.
pub const RegionDescriptor = struct {
    name: []const u8,
    annotation: pr.Annotation,

    /// Equations in this region (slice of Function.eqns).
    eqns: []const pr.Eqn,

    /// Full aval table from the parent function (needed for type lookup).
    avals: []const pr.Aval,

    /// Backing store for equation input/output var IDs.
    varids_store: []const pr.VarId,

    /// Backing store for equation parameters.
    params_store: []const pr.Param,

    /// VarIds produced outside this region but consumed inside it.
    inputs: []const pr.VarId,

    /// VarIds produced inside this region and consumed outside it.
    outputs: []const pr.VarId,

    /// Look up the type of a variable.
    pub fn aval_of(self: RegionDescriptor, id: pr.VarId) ?pr.Aval {
        const idx: usize = @intCast(id);
        if (idx >= self.avals.len) return null;
        return self.avals[idx];
    }

    /// Get a human-readable summary for diagnostics.
    pub fn eqn_count(self: RegionDescriptor) usize {
        return self.eqns.len;
    }
};

/// Build a RegionDescriptor from a Function and a Region.
///
/// Computes the boundary variables (inputs/outputs) by analyzing which
/// VarIds are produced/consumed inside vs outside the region.
pub fn describe_region(allocator: std.mem.Allocator, func: pr.Function, region: pr.Region) error{OutOfMemory}!RegionDescriptor {
    const start: usize = region.eqn_start;
    const len: usize = region.eqn_len;
    const end = start + len;
    if (end > func.eqns.len) return error.OutOfMemory; // bounds check

    const region_eqns = func.eqns[start..end];

    // Collect VarIds produced inside the region.
    var produced = std.AutoHashMap(pr.VarId, void).init(allocator);
    defer produced.deinit();
    for (region_eqns) |eqn| {
        const outs = eqn.outputs.slice(pr.VarId, func.varids_store);
        for (outs) |out_id| try produced.put(out_id, {});
    }

    // Inputs: consumed inside but not produced inside.
    var inputs_list = std.ArrayList(pr.VarId).empty;
    defer inputs_list.deinit(allocator);
    var seen_inputs = std.AutoHashMap(pr.VarId, void).init(allocator);
    defer seen_inputs.deinit();
    for (region_eqns) |eqn| {
        const ins = eqn.inputs.slice(pr.VarId, func.varids_store);
        for (ins) |in_id| {
            if (!produced.contains(in_id) and !seen_inputs.contains(in_id)) {
                try inputs_list.append(allocator, in_id);
                try seen_inputs.put(in_id, {});
            }
        }
    }

    // Outputs: produced inside and consumed outside (or is a function return).
    var consumed_outside = std.AutoHashMap(pr.VarId, void).init(allocator);
    defer consumed_outside.deinit();

    // Check equations outside the region for consumption.
    for (func.eqns, 0..) |eqn, idx| {
        if (idx >= start and idx < end) continue;
        const ins = eqn.inputs.slice(pr.VarId, func.varids_store);
        for (ins) |in_id| {
            if (produced.contains(in_id)) try consumed_outside.put(in_id, {});
        }
    }

    // Also count function returns as external consumption.
    for (func.returns) |ret_id| {
        if (produced.contains(ret_id)) try consumed_outside.put(ret_id, {});
    }

    var outputs_list = std.ArrayList(pr.VarId).empty;
    defer outputs_list.deinit(allocator);
    // Preserve output order by iterating region eqns.
    var seen_outputs = std.AutoHashMap(pr.VarId, void).init(allocator);
    defer seen_outputs.deinit();
    for (region_eqns) |eqn| {
        const outs = eqn.outputs.slice(pr.VarId, func.varids_store);
        for (outs) |out_id| {
            if (consumed_outside.contains(out_id) and !seen_outputs.contains(out_id)) {
                try outputs_list.append(allocator, out_id);
                try seen_outputs.put(out_id, {});
            }
        }
    }

    return .{
        .name = region.name,
        .annotation = region.annotation,
        .eqns = region_eqns,
        .avals = func.avals,
        .varids_store = func.varids_store,
        .params_store = func.params_store,
        .inputs = try inputs_list.toOwnedSlice(allocator),
        .outputs = try outputs_list.toOwnedSlice(allocator),
    };
}

// ============================================================================
// Dispatch Types
// ============================================================================

/// Element data type for dispatch buffers.
///
/// Subset of types relevant to kernel dispatch. Providers validate
/// dtype support internally; the backend maps from FFI-layer types.
pub const DType = enum {
    f16,
    bf16,
    f32,
    f64,
    i8,
    i32,
    i64,
    u32,
    u64,
};

/// Execution platform for dispatch.
pub const DispatchPlatform = enum { host, cuda };

/// Descriptor for a single buffer passed through the FFI boundary.
///
/// Provider-agnostic: the backend extracts these from FFI frames,
/// providers consume them without knowing about XLA types.
pub const BufferDesc = struct {
    data: *anyopaque,
    dtype: DType,
    dims: []const i64,
    rank: usize,
};

/// Context passed to a provider's dispatch function at execution time.
///
/// Contains all information a provider needs to execute a compiled kernel:
/// input/output buffers, device identity, and an optional device stream
/// for GPU synchronization.
pub const DispatchContext = struct {
    inputs: []const BufferDesc,
    outputs: []const BufferDesc,
    device_ordinal: i32,
    platform: DispatchPlatform,
    /// GPU stream handle (e.g. CUDA stream). Null on host.
    stream: ?*anyopaque,
    allocator: std.mem.Allocator,
};

pub const DispatchError = error{ DispatchFailed, UnsupportedDType, ShapeMismatch, OutOfMemory };

/// Provider dispatch function signature.
///
/// Called by the backend's generic FFI handler when a custom_call
/// targets a kernelized op. The provider_ctx is the provider's own
/// state (cast from `*anyopaque`); artifact_data and kernel_key
/// identify the compiled kernel; ctx carries buffers and device info.
pub const DispatchFn = *const fn (
    provider_ctx: *anyopaque,
    artifact_data: []const u8,
    kernel_key: []const u8,
    ctx: DispatchContext,
) DispatchError!void;

// ============================================================================
// Kernel Artifact
// ============================================================================

/// A compiled kernel for a specific target.
///
/// Produced by a kernel provider, consumed by the backend at execution
/// time. The dispatch_fn + dispatch_ctx pair allow the backend to call
/// into the provider without knowing its identity.
pub const KernelArtifact = struct {
    /// Provider that produced this artifact.
    provider_name: []const u8,

    /// Opaque compiled kernel data (e.g. .so bytes).
    /// Ownership is transferred to the KernelRegistry; must be allocated
    /// with the allocator passed to provider.compile.
    data: []const u8,

    /// Target name used to reference this KA from custom_call ops.
    target_name: []const u8,

    /// Provider dispatch entry point.
    dispatch_fn: ?DispatchFn = null,

    /// Provider-owned state passed as first argument to dispatch_fn.
    /// Non-owning: must outlive all KernelArtifacts that reference it.
    dispatch_ctx: ?*anyopaque = null,

    /// Execute this kernel artifact via its provider's dispatch function.
    ///
    /// The caller supplies the kernel_key used for lookup (from the
    /// custom_call attributes). This may differ from target_name if
    /// aliasing or versioned keys are in play.
    pub fn dispatch(self: *const KernelArtifact, kernel_key: []const u8, ctx: DispatchContext) DispatchError!void {
        const dfn = self.dispatch_fn orelse return error.DispatchFailed;
        const dctx = self.dispatch_ctx orelse return error.DispatchFailed;
        return dfn(dctx, self.data, kernel_key, ctx);
    }

    pub fn deinit(self: *KernelArtifact, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

// ============================================================================
// Kernel Provider
// ============================================================================

pub const CompileError = error{
    /// Provider cannot handle this region (unsupported ops, shapes, etc.)
    Unsupported,
    /// Compilation failed for an internal reason.
    CompileFailed,
    OutOfMemory,
};

/// Extension component that can claim PR regions and produce KAs.
///
/// Follows the Zig interface pattern (ptr + function pointer).
/// Implementations provide a compile function; the core system
/// calls it via the `compile` method.
pub const KernelProvider = struct {
    name: []const u8,
    ptr: *anyopaque,
    compile_fn: *const fn (ptr: *anyopaque, desc: RegionDescriptor, allocator: std.mem.Allocator) CompileError!KernelArtifact,

    pub fn compile(self: KernelProvider, desc: RegionDescriptor, allocator: std.mem.Allocator) CompileError!KernelArtifact {
        return self.compile_fn(self.ptr, desc, allocator);
    }
};

// ============================================================================
// Kernel Registry
// ============================================================================

/// Maps custom_call target names to compiled kernel artifacts.
///
/// The kernelization pass writes entries. The backend reads entries
/// when resolving custom_call references at execution time.
/// Owned by the coordinator (user code).
pub const KernelRegistry = struct {
    entries: std.StringHashMap(KernelArtifact),

    pub fn init(allocator: std.mem.Allocator) KernelRegistry {
        return .{ .entries = std.StringHashMap(KernelArtifact).init(allocator) };
    }

    pub fn allocator(self: *const KernelRegistry) std.mem.Allocator {
        return self.entries.allocator;
    }

    pub fn deinit(self: *KernelRegistry) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            var artifact = entry.value_ptr.*;
            artifact.deinit(self.entries.allocator);
            self.entries.allocator.free(entry.key_ptr.*);
        }
        self.entries.deinit();
    }

    pub fn put(self: *KernelRegistry, target_name: []const u8, artifact: KernelArtifact) error{OutOfMemory, DuplicateKey}!void {
        if (self.entries.contains(target_name)) return error.DuplicateKey;
        const owned_key = try self.entries.allocator.dupe(u8, target_name);
        errdefer self.entries.allocator.free(owned_key);
        try self.entries.put(owned_key, artifact);
    }

    pub fn get(self: *const KernelRegistry, target_name: []const u8) ?KernelArtifact {
        return self.entries.get(target_name);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "describe_region computes boundary vars" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    // Build: x, y are params. Region covers z = x + y. Output z is returned.
    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    try b.push_region("add_region", .{ .kernelize = "test" });
    const z = try b.emit(.add, &.{ x, y }, &.{});
    try b.pop_region();

    const func = try b.finish(&.{z});
    try program.add_function(func);

    try testing.expectEqual(@as(usize, 1), func.regions.len);

    const desc = try describe_region(testing.allocator, func, func.regions[0]);
    defer testing.allocator.free(desc.inputs);
    defer testing.allocator.free(desc.outputs);

    try testing.expectEqualStrings("add_region", desc.name);
    try testing.expectEqual(@as(usize, 1), desc.eqns.len);
    try testing.expectEqual(@as(usize, 2), desc.inputs.len); // x, y
    try testing.expectEqual(@as(usize, 1), desc.outputs.len); // z
    try testing.expectEqual(x, desc.inputs[0]);
    try testing.expectEqual(y, desc.inputs[1]);
    try testing.expectEqual(z, desc.outputs[0]);
}

test "describe_region internal vars not in outputs" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("chain", .{ .kernelize = "test" });
    const tmp = try b.emit(.exp, &.{x}, &.{});
    const out = try b.emit(.log, &.{tmp}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{out});
    try program.add_function(func);

    const desc = try describe_region(testing.allocator, func, func.regions[0]);
    defer testing.allocator.free(desc.inputs);
    defer testing.allocator.free(desc.outputs);

    try testing.expectEqual(@as(usize, 2), desc.eqns.len);
    try testing.expectEqual(@as(usize, 1), desc.inputs.len); // x
    try testing.expectEqual(@as(usize, 1), desc.outputs.len); // out (tmp is internal)
}

test "kernel registry put and get" {
    const testing = std.testing;

    var registry = KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    const artifact = KernelArtifact{
        .provider_name = "test",
        .data = try testing.allocator.dupe(u8, "fake_kernel"),
        .target_name = "zigrad.kernel.test_0",
    };
    try registry.put("zigrad.kernel.test_0", artifact);

    const found = registry.get("zigrad.kernel.test_0");
    try testing.expect(found != null);
    try testing.expectEqualStrings("fake_kernel", found.?.data);

    try testing.expect(registry.get("nonexistent") == null);
}
