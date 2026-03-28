/// Kernel Provider Interface and Store Types
///
/// Type hierarchy:
/// - `KernelProvider`: extension point -- compiles regions into `KernelArtifact`s.
/// - `KernelArtifact`: pure compile output -- data, provider name, workspace.
/// - `KernelStore`: pre-computed tuning decisions keyed by shape signature.
/// - `DispatchRegistry`: maps provider names to dispatch function pointers.
/// - `StoredArtifact` / `Decision`: store-internal value types.
///
/// Data flow: `tune()` invokes providers -> populates `KernelStore` +
/// `DispatchRegistry` -> pipeline consults store -> backend resolves dispatch
/// from registry at execute time.
///
/// This module is independent of the pipeline and backend. It depends only
/// on PR types.
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
/// RegionDescriptor does not own any memory -- all slices are views into
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

/// Returns whether a dot_general parameter set matches plain rank-2 matmul.
///
/// Accepts only no batch dimensions and one contracting dimension per input,
/// with lhs contracting dim 1 and rhs contracting dim 0.
pub fn dot_general_is_matrix_matmul(params: []const pr.Param) bool {
    return dot_general_is_canonical_batched_matmul(params, 2, 2);
}

/// Returns whether dot_general matches canonical batched matmul form.
///
/// Canonical form requires both operands to have rank `batch_len + 2` with
/// batch dims as `0..batch_len-1`, lhs contracting dim at `rank-1`, and rhs
/// contracting dim at `rank-2`.
pub fn dot_general_is_canonical_batched_matmul(params: []const pr.Param, lhs_rank: usize, rhs_rank: usize) bool {
    const dg = pr.param(.dot_general, params) orelse return false;
    const batch_len = dg.lhs_batch_dims.len;

    if (batch_len != dg.rhs_batch_dims.len) return false;
    if (dg.lhs_contracting_dims.len != 1 or dg.rhs_contracting_dims.len != 1) return false;
    if (lhs_rank != rhs_rank) return false;
    if (lhs_rank != batch_len + 2) return false;
    if (!dims_are_prefix(dg.lhs_batch_dims) or !dims_are_prefix(dg.rhs_batch_dims)) return false;

    const lhs_contract_expected: i64 = @intCast(lhs_rank - 1);
    const rhs_contract_expected: i64 = @intCast(rhs_rank - 2);
    return dg.lhs_contracting_dims[0] == lhs_contract_expected and dg.rhs_contracting_dims[0] == rhs_contract_expected;
}

fn dims_are_prefix(dims: []const i64) bool {
    for (dims, 0..) |dim, idx| {
        if (dim != @as(i64, @intCast(idx))) return false;
    }
    return true;
}

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
/// TODO: Fix this re-export pattern
pub const DType = pr.DType;

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
    /// Optional provider workspace pointer.
    workspace: ?*anyopaque,
    /// Compile-time workspace requirement reported by the provider artifact.
    workspace_bytes_required: usize,
    allocator: std.mem.Allocator,
};

pub const DispatchError = error{
    /// Dispatch failed; provider logged details.
    DispatchFailed,
    UnsupportedDType,
    ShapeMismatch,
    /// Provider runtime not available (library loading failed).
    ProviderLoadFailed,
    WorkspaceUnavailable,
    OutOfMemory,
};

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

/// Pure compile output from a kernel provider.
///
/// Contains the provider name, compiled data, target name, and workspace
/// requirement. Carries no runtime pointers -- dispatch is resolved at
/// execute time via `DispatchRegistry` (for store-based paths) or via
/// `KernelProvider.dispatch_fn` (registered per-provider, not per-artifact).
pub const KernelArtifact = struct {
    /// Provider that produced this artifact.
    provider_name: []const u8,

    /// Opaque compiled kernel data (e.g. .so bytes).
    /// Ownership depends on context: `tune()` copies into the store.
    data: []const u8,

    /// Target name used to reference this KA from custom_call ops.
    target_name: []const u8,

    /// Workspace bytes required at dispatch time.
    ///
    /// Providers set this from compile metadata; the backend allocates
    /// device memory of this size before calling the dispatch function.
    workspace_bytes: usize = 0,

    pub fn deinit(self: *KernelArtifact, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.target_name);
        self.* = undefined;
    }
};

// ============================================================================
// Kernel Provider
// ============================================================================

// ============================================================================
// Compile Context
// ============================================================================

/// Device targeting context for provider compilation.
///
/// Passed by `tune()` to providers so they can target the correct device
/// architecture. Extensible via defaulted fields -- add new targeting
/// knobs here without breaking existing providers.
pub const CompileContext = struct {
    device_ordinal: i32 = 0,
    platform: DispatchPlatform = .host,
};

// ============================================================================
// Compile Errors and Provider
// ============================================================================

pub const CompileError = error{
    /// Provider cannot handle this region (unsupported ops, shapes, etc.)
    Unsupported,
    /// Compilation failed; provider logged details.
    CompileFailed,
    /// Provider runtime not available (library loading failed).
    ProviderLoadFailed,
    /// Provider API call failed or returned unexpected data.
    ProviderCallFailed,
    OutOfMemory,
};

/// Extension component that claims PR regions and produces compiled kernel artifacts.
///
/// Providers implement `compile_fn` (and optionally `finalize_fn`).
/// The compile function receives a `CompileContext` for device targeting.
///
/// Dispatch fields (`dispatch_fn`, `dispatch_ctx`) identify how to execute compiled
/// artifacts at runtime. They are per-provider (not per-artifact) because all artifacts
/// from a given provider share the same dispatch implementation and state.
pub const KernelProvider = struct {
    name: []const u8,
    ptr: *anyopaque,
    compile_fn: *const fn (ptr: *anyopaque, desc: RegionDescriptor, ctx: CompileContext, allocator: std.mem.Allocator) CompileError!KernelArtifact,
    finalize_fn: ?*const fn (ptr: *anyopaque) void = null,
    /// Provider's dispatch function. Called by the backend's FFI handler at execute time.
    /// Must be set for providers whose artifacts require runtime dispatch.
    dispatch_fn: ?DispatchFn = null,
    /// Provider-owned state passed as first arg to `dispatch_fn`.
    /// Must outlive all executions that reference this provider.
    dispatch_ctx: ?*anyopaque = null,

    pub fn compile(self: KernelProvider, desc: RegionDescriptor, ctx: CompileContext, allocator: std.mem.Allocator) CompileError!KernelArtifact {
        return self.compile_fn(self.ptr, desc, ctx, allocator);
    }

    /// Release provider resources after all kernels have been compiled.
    ///
    /// Providers set this to free heavyweight state (e.g. GPU memory pools)
    /// that would otherwise compete with the backend allocator. No-op when
    /// the provider leaves `finalize_fn` as `null`.
    pub fn finalize(self: KernelProvider) void {
        const f = self.finalize_fn orelse return;
        f(self.ptr);
    }
};

// ============================================================================
// Kernel Store (Decision Boundary)
// ============================================================================

/// A pre-compiled artifact stored in the kernel store.
///
/// Pure data -- dispatch is resolved at execute time via `DispatchRegistry`.
/// The store owns copies of all slices (`provider_name`, `data`, `target_name`).
pub const StoredArtifact = struct {
    provider_name: []const u8,
    data: []const u8,
    target_name: []const u8,
    workspace_bytes: usize = 0,
};

/// Result of a tuning decision for a given kernel key.
///
/// `.profitable`: provider compiled a kernel successfully and the artifact is stored.
/// `.negative`: provider evaluated the region and decided not to kernelize (reason recorded).
/// Absence from the store (null from `get()`) means the key was never evaluated --
/// the region should be left for baseline lowering.
pub const Decision = union(enum) {
    profitable: StoredArtifact,
    negative: []const u8,
};

/// Pre-computed tuning decisions keyed by kernel signature.
///
/// The sole decision boundary between tuning and the pipeline. `tune()`
/// populates the store; `KernelizePass` consults it. The pipeline never
/// invokes providers directly -- all provider interaction happens in `tune()`.
pub const KernelStore = struct {
    decisions: std.StringHashMap(Decision),

    pub fn init(store_allocator: std.mem.Allocator) KernelStore {
        return .{ .decisions = std.StringHashMap(Decision).init(store_allocator) };
    }

    pub fn allocator(self: *const KernelStore) std.mem.Allocator {
        return self.decisions.allocator;
    }

    pub fn deinit(self: *KernelStore) void {
        var it = self.decisions.iterator();
        while (it.next()) |entry| {
            switch (entry.value_ptr.*) {
                .profitable => |art| {
                    self.decisions.allocator.free(art.data);
                    self.decisions.allocator.free(art.target_name);
                    self.decisions.allocator.free(art.provider_name);
                },
                .negative => |reason| {
                    self.decisions.allocator.free(reason);
                },
            }
            self.decisions.allocator.free(entry.key_ptr.*);
        }
        self.decisions.deinit();
    }

    /// Record a profitable tuning decision.
    pub fn put_profitable(self: *KernelStore, kernel_signature: []const u8, artifact: StoredArtifact) error{OutOfMemory}!void {
        const owned_key = try self.decisions.allocator.dupe(u8, kernel_signature);
        errdefer self.decisions.allocator.free(owned_key);
        const owned_name = try self.decisions.allocator.dupe(u8, artifact.provider_name);
        errdefer self.decisions.allocator.free(owned_name);
        const owned_data = try self.decisions.allocator.dupe(u8, artifact.data);
        errdefer self.decisions.allocator.free(owned_data);
        const owned_target = try self.decisions.allocator.dupe(u8, artifact.target_name);
        errdefer self.decisions.allocator.free(owned_target);

        try self.decisions.put(owned_key, .{ .profitable = .{
            .provider_name = owned_name,
            .data = owned_data,
            .target_name = owned_target,
            .workspace_bytes = artifact.workspace_bytes,
        } });
    }

    /// Record a negative tuning decision (provider declined).
    pub fn put_negative(self: *KernelStore, kernel_signature: []const u8, reason: []const u8) error{OutOfMemory}!void {
        const owned_key = try self.decisions.allocator.dupe(u8, kernel_signature);
        errdefer self.decisions.allocator.free(owned_key);
        const owned_reason = try self.decisions.allocator.dupe(u8, reason);
        errdefer self.decisions.allocator.free(owned_reason);
        try self.decisions.put(owned_key, .{ .negative = owned_reason });
    }

    /// Look up a tuning decision. Returns null if the key was never evaluated.
    pub fn get(self: *const KernelStore, kernel_signature: []const u8) ?Decision {
        return self.decisions.get(kernel_signature);
    }

    /// Returns true if the key has a profitable decision.
    pub fn is_profitable(self: *const KernelStore, kernel_signature: []const u8) bool {
        const decision = self.decisions.get(kernel_signature) orelse return false;
        return switch (decision) {
            .profitable => true,
            .negative => false,
        };
    }
};

// ============================================================================
// Dispatch Registry
// ============================================================================

/// Entry mapping a provider name to its dispatch function and context.
///
/// Populated at execute time by the caller. The backend resolves provider
/// names from `StoredArtifact.provider_name` to dispatch entries here.
pub const DispatchEntry = struct {
    dispatch_fn: DispatchFn,
    dispatch_ctx: *anyopaque,
};

/// Maps provider names to dispatch entries for execute-time resolution.
///
/// Populated before execution begins (typically by `tune()`), consulted at
/// dispatch time by the backend's FFI handler. Decouples compile-time
/// decisions (in the store) from execute-time dispatch (function pointers).
pub const DispatchRegistry = struct {
    entries: std.StringHashMap(DispatchEntry),

    pub fn init(reg_allocator: std.mem.Allocator) DispatchRegistry {
        return .{ .entries = std.StringHashMap(DispatchEntry).init(reg_allocator) };
    }

    pub fn deinit(self: *DispatchRegistry) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.entries.allocator.free(entry.key_ptr.*);
        }
        self.entries.deinit();
    }

    /// Register a provider's dispatch entry. Duplicates are silently replaced.
    pub fn register(self: *DispatchRegistry, provider_name: []const u8, entry: DispatchEntry) error{OutOfMemory}!void {
        const result = self.entries.getOrPut(provider_name) catch return error.OutOfMemory;
        if (!result.found_existing) {
            result.key_ptr.* = try self.entries.allocator.dupe(u8, provider_name);
        }
        result.value_ptr.* = entry;
    }

    /// Look up a dispatch entry by provider name.
    pub fn get(self: *const DispatchRegistry, provider_name: []const u8) ?DispatchEntry {
        return self.entries.get(provider_name);
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

test "dot_general_is_matrix_matmul canonical" {
    const params = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{0},
    } }};
    try std.testing.expect(dot_general_is_matrix_matmul(params[0..]));
}

test "dot_general_is_matrix_matmul rejects non-canonical" {
    const with_batch = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{0},
    } }};
    try std.testing.expect(!dot_general_is_matrix_matmul(with_batch[0..]));

    const wrong_contract = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{0},
        .rhs_contracting_dims = &.{1},
    } }};
    try std.testing.expect(!dot_general_is_matrix_matmul(wrong_contract[0..]));
}

test "dot_general_is_canonical_batched_matmul canonical" {
    const params = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    } }};
    try std.testing.expect(dot_general_is_canonical_batched_matmul(params[0..], 4, 4));
}

test "dot_general_is_canonical_batched_matmul rejects non-prefix batch" {
    const params = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{1},
        .rhs_batch_dims = &.{1},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    } }};
    try std.testing.expect(!dot_general_is_canonical_batched_matmul(params[0..], 3, 3));
}

test "dot_general_is_canonical_batched_matmul rejects rank mismatch" {
    const params = [_]pr.Param{.{ .dot_general = .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    } }};
    try std.testing.expect(!dot_general_is_canonical_batched_matmul(params[0..], 3, 4));
}

test "finalize calls hook when set" {
    const Hook = struct {
        called: bool = false,
        fn impl(ptr: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.called = true;
        }
    };
    var hook = Hook{};
    const provider = KernelProvider{
        .name = "test",
        .ptr = @ptrCast(&hook),
        .compile_fn = undefined,
        .finalize_fn = Hook.impl,
    };
    provider.finalize();
    try std.testing.expect(hook.called);
}

test "finalize is no-op when null" {
    const provider = KernelProvider{
        .name = "test",
        .ptr = undefined,
        .compile_fn = undefined,
    };
    // finalize_fn defaults to null; calling finalize must not panic.
    provider.finalize();
}

test "kernel store put and get profitable" {
    const testing = std.testing;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try store.put_profitable("matmul_f32_128x128", .{
        .provider_name = "mirage",
        .data = "compiled_kernel_bytes",
        .target_name = "zigrad.kernel.matmul_0",
        .workspace_bytes = 4096,
    });

    const decision = store.get("matmul_f32_128x128") orelse return error.TestUnexpectedResult;
    switch (decision) {
        .profitable => |art| {
            try testing.expectEqualStrings("mirage", art.provider_name);
            try testing.expectEqualStrings("compiled_kernel_bytes", art.data);
            try testing.expectEqualStrings("zigrad.kernel.matmul_0", art.target_name);
            try testing.expectEqual(@as(usize, 4096), art.workspace_bytes);
        },
        .negative => return error.TestUnexpectedResult,
    }
    try testing.expect(store.is_profitable("matmul_f32_128x128"));
}

test "kernel store put and get negative" {
    const testing = std.testing;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try store.put_negative("conv_f32_3x3", "unsupported shape");

    const decision = store.get("conv_f32_3x3") orelse return error.TestUnexpectedResult;
    switch (decision) {
        .profitable => return error.TestUnexpectedResult,
        .negative => |reason| {
            try testing.expectEqualStrings("unsupported shape", reason);
        },
    }
    try testing.expect(!store.is_profitable("conv_f32_3x3"));
}

test "kernel store absent key" {
    const testing = std.testing;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try testing.expect(store.get("nonexistent") == null);
    try testing.expect(!store.is_profitable("nonexistent"));
}

test "dispatch registry register and get" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: *anyopaque, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var dummy_ctx: u8 = 42;
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = @ptrCast(&dummy_ctx),
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expect(entry.dispatch_ctx == @as(*anyopaque, @ptrCast(&dummy_ctx)));
    try testing.expect(registry.get("tvm") == null);
}

test "dispatch registry replaces duplicate" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: *anyopaque, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var ctx1: u8 = 1;
    var ctx2: u8 = 2;
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = @ptrCast(&ctx1),
    });
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = @ptrCast(&ctx2),
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expect(entry.dispatch_ctx == @as(*anyopaque, @ptrCast(&ctx2)));
}
