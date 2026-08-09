//! Kernel-provider decisions, artifacts, and dispatch registration.
//!
//! Tuning asks providers to compile callable PR functions and records their decisions in a
//!  `KernelStore`.
//!
//! Kernelization reads the store, and execution resolves providers through a
//!  separate `DispatchRegistry`.
const std = @import("std");
const device = @import("../device.zig");
const fingerprint = @import("fingerprint.zig");
const pr = @import("pr.zig");
const TypedPtr = @import("../utils/rtti.zig").TypedPtr;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.@"zg/kernel");

/// Custom-call target used by kernel-provider dispatch.
pub const dispatch_target_name = "zigrad.kernel.dispatch";

/// Region annotation requesting a candidate from a named kernel provider.
pub const provider_annotation_name = "zigrad.kernel.provider";

/// Invalid payloads for the provider annotation contract.
pub const AnnotationError = error{
    InvalidProviderAnnotation,
    ProviderRegionNotOutlined,
};

/// Construct a provider request for a region builder.
pub fn provider_annotation(provider_name: []const u8) pr.Annotation {
    return .{
        .name = provider_annotation_name,
        .value = .{ .string = provider_name },
    };
}

/// Return the provider requested by an annotated IR object.
pub fn requested_provider(owner: anytype) AnnotationError!?[]const u8 {
    const found = owner.find_annotation(provider_annotation_name) orelse return null;
    return found.value.as_string() orelse error.InvalidProviderAnnotation;
}

/// Reject provider requests that have not been outlined into functions.
pub fn require_outlined_requests(program: *const pr.Program) AnnotationError!void {
    for (program.functions) |func| {
        for (func.regions) |region| {
            if (try requested_provider(region) != null) {
                return error.ProviderRegionNotOutlined;
            }
        }
    }
}

/// Returns whether a dot_general parameter set matches plain rank-2 matmul.
///
/// Accepts only no batch dimensions and one contracting dimension per input,
///  with lhs contracting dim 1 and rhs contracting dim 0.
pub fn dot_general_is_matrix_matmul(dg: pr.DotGeneralParams) bool {
    return dot_general_is_canonical_batched_matmul(dg, 2, 2);
}

/// Returns whether dot_general matches canonical batched matmul form.
///
/// Canonical form requires both operands to have rank `batch_len + 2` with
///  batch dims as `0..batch_len-1`, lhs contracting dim at `rank-1`, and rhs
///  contracting dim at `rank-2`.
pub fn dot_general_is_canonical_batched_matmul(dg: pr.DotGeneralParams, lhs_rank: usize, rhs_rank: usize) bool {
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

/// Store key for one provider decision on one device.
pub const DecisionKey = struct {
    bytes: []const u8,
};

/// Failures produced while encoding decision identity.
pub const IdentityError = Allocator.Error || std.Io.Writer.Error;

/// Combine a requested provider, selected device, and function fingerprint.
///
/// Platform names are normalized because `Platform.eql` ignores ASCII case.
/// Caller owns result.
pub fn make_decision_key(
    allocator: Allocator,
    provider_name: []const u8,
    selected_device: device.Device,
    function_fingerprint: fingerprint.Function,
) IdentityError!DecisionKey {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    const writer = &output.writer;

    try writer.print("kp2:{d}:", .{provider_name.len});
    try writer.writeAll(provider_name);
    try writer.print(":{d}:", .{selected_device.platform.name.len});
    for (selected_device.platform.name) |byte| {
        try writer.writeByte(std.ascii.toLower(byte));
    }
    try writer.print(":{d}:", .{selected_device.ordinal});
    try function_fingerprint.write_hex(writer);

    return .{ .bytes = try output.toOwnedSlice() };
}

/// Append one abstract value to a diagnostic string.
pub fn write_aval_signature(writer: anytype, aval: pr.Aval) !void {
    switch (aval) {
        .tensor => |tensor| {
            try writer.writeAll(@tagName(tensor.dtype));
            try writer.writeByte('[');
            for (tensor.shape.dims, 0..) |dim, index| {
                if (index > 0) try writer.writeByte(',');
                try writer.print("{d}", .{dim});
            }
            try writer.writeByte(']');
        },
    }
}

/// Element data type for dispatch buffers.
pub const DType = @import("../dtype.zig").DType;

/// Descriptor for a single buffer passed through an FFI call.
///
/// The execution integration extracts these descriptors from FFI frames.
pub const BufferDesc = struct {
    data: *anyopaque,
    dtype: DType,
    dims: []const i64,
    rank: usize,
};

/// Context passed to a provider's dispatch function at execution time.
///
/// It supplies buffers, device identity, an optional device stream, and workspace.
///
/// TODO(kernel-provider): Pass a device allocator when integrations expose one.
pub const DispatchContext = struct {
    inputs: []const BufferDesc,
    outputs: []const BufferDesc,
    device: device.Device,

    /// Platform stream handle received from the integration FFI, or null on host.
    stream: ?*anyopaque = null,

    /// Provider workspace, or null when the artifact requires none.
    workspace: ?*anyopaque = null,

    /// Compile-time workspace requirement reported by the provider artifact.
    workspace_bytes_required: usize,

    /// Allocator available for host-side dispatch work.
    allocator: std.mem.Allocator,
};

/// Failures exposed by kernel-provider dispatch.
pub const DispatchError = error{
    /// Dispatch failed, provider logged details.
    DispatchFailed,
    UnsupportedDType,
    ShapeMismatch,
    UnsupportedDevice,
    /// Provider runtime not available (library loading failed).
    ProviderLoadFailed,
    WorkspaceUnavailable,
} || Allocator.Error;

/// Context available while preparing provider artifacts for execution.
pub const PrepareContext = struct {
    /// Device selected for the upcoming execution.
    device: device.Device,
};

/// Failures exposed while preparing kernel artifacts for execution.
pub const PrepareError = error{
    /// The store references a provider absent from the dispatch registry.
    ProviderNotRegistered,
    /// Artifact preparation failed, provider logged details.
    PrepareFailed,
    /// Provider runtime not available.
    ProviderLoadFailed,
    UnsupportedDevice,
} || Allocator.Error;

/// Provider artifact-preparation function signature.
///
/// Preparation creates process-local runtime state from portable artifact
///  bytes. Callers invoke it before execution begins.
pub const PrepareFn = *const fn (
    provider_ctx: TypedPtr,
    artifact_data: []const u8,
    kernel_key: []const u8,
    ctx: PrepareContext,
) PrepareError!void;

/// Provider dispatch function signature.
///
/// Called by an execution integration when a custom call targets a kernelized op.
///
/// `provider_ctx` contains the provider state and carries its type through
///  `TypedPtr`.
pub const DispatchFn = *const fn (
    provider_ctx: TypedPtr,
    /// Opaque provider artifact bytes.
    artifact_data: []const u8,
    /// Key to identify the compiled kernel.
    kernel_key: []const u8,
    ctx: DispatchContext,
) DispatchError!void;

/// Portable kernel-provider artifact.
///
/// It contains no runtime pointers. The dispatch registry resolves the provider
///  when execution reaches the corresponding custom call.
pub const Artifact = struct {
    /// Opaque kernel data allocated with the provider compilation allocator.
    data: []const u8,

    /// Workspace bytes required at dispatch time.
    ///
    /// Providers set this from compile metadata, and the execution integration
    ///  allocates device memory before calling the dispatch function.
    workspace_bytes: usize = 0,

    /// Byte alignment required for the workspace allocation.
    ///
    /// This value must be a nonzero power of two when `workspace_bytes` is
    ///  nonzero.
    workspace_alignment: usize = 1,

    /// Release artifact data with the provider compilation allocator.
    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

/// Profitable provider decision and its portable artifact.
pub const ProfitableDecision = struct {
    /// Provider selected for this decision.
    provider_name: []const u8,

    /// Artifact returned by the selected provider.
    artifact: Artifact,
};

/// Failures exposed by kernel-provider compilation.
pub const CompileError = error{
    /// Provider cannot handle this function.
    Unsupported,

    /// Compilation failed, provider logged details.
    CompileFailed,

    /// Provider runtime not available (library loading failed).
    ProviderLoadFailed,

    /// Provider API call failed or returned unexpected data.
    ProviderCallFailed,
} || Allocator.Error;

/// Extension component that claims PR functions and produces compiled kernel artifacts.
///
/// `compile_fn` receives the selected device for target resolution.
///
/// Runtime hooks share one context across every artifact from the provider.
pub const KernelProvider = struct {
    /// Stable provider name used in decision and dispatch keys.
    name: []const u8,

    /// Provider state passed to `compile_fn` and `finalize_fn`.
    ptr: *anyopaque,

    /// Compile one callable PR function for the selected device.
    compile_fn: *const fn (ptr: *anyopaque, func: pr.Function, selected_device: device.Device, allocator: std.mem.Allocator) CompileError!Artifact,

    /// Optional release hook for compilation-only provider resources.
    finalize_fn: ?*const fn (ptr: *anyopaque) void = null,

    /// Provider dispatch function called by the execution integration.
    ///
    /// Must be set for providers whose artifacts require runtime dispatch.
    dispatch_fn: ?DispatchFn = null,

    /// Optional hook that prepares artifacts before execution.
    prepare_fn: ?PrepareFn = null,

    /// Provider state passed to the runtime hooks.
    ///
    /// The type tag lets each runtime hook check the concrete state type,
    ///  and the state must outlive every execution that references this provider.
    dispatch_ctx: ?TypedPtr = null,

    /// Compile one callable PR function for the selected device.
    pub fn compile(self: KernelProvider, func: pr.Function, selected_device: device.Device, allocator: std.mem.Allocator) CompileError!Artifact {
        return try self.compile_fn(self.ptr, func, selected_device, allocator);
    }

    /// Release compilation-only provider resources.
    ///
    /// Providers use this hook before execution starts. Calling it without a
    ///  hook has no effect.
    pub fn finalize(self: KernelProvider) void {
        const f = self.finalize_fn orelse return;
        f(self.ptr);
    }
};

/// Artifact-data handling for `KernelStore.put_profitable`.
pub const ArtifactStorage = enum {
    /// Copy artifact data into the store.
    copy,

    /// Transfer artifact data allocated by the store allocator.
    ///
    /// The call consumes `artifact.data`, including when insertion fails.
    take,
};

/// Result of one provider decision.
///
/// Absence from the store means the provider did not evaluate the key.
pub const Decision = union(enum) {
    /// Provider supplied an artifact for this key.
    profitable: ProfitableDecision,

    /// Provider declined this key with the recorded reason.
    negative: []const u8,
};

/// Pre-computed tuning decisions keyed by provider, device, and function fingerprint.
///
/// The sole decision source used by kernelization.
///
/// `tune()` populates the store. `KernelizePass` consults it and never invokes
///  providers directly.
pub const KernelStore = struct {
    decisions: std.StringHashMap(Decision),

    /// Initialize an empty decision store.
    pub fn init(store_allocator: std.mem.Allocator) KernelStore {
        return .{ .decisions = std.StringHashMap(Decision).init(store_allocator) };
    }

    /// Return the allocator used by this store.
    pub fn allocator(self: *const KernelStore) std.mem.Allocator {
        return self.decisions.allocator;
    }

    /// Release every stored decision allocation.
    pub fn deinit(self: *KernelStore) void {
        var it = self.decisions.iterator();
        while (it.next()) |entry| {
            switch (entry.value_ptr.*) {
                .profitable => |stored| {
                    self.decisions.allocator.free(stored.artifact.data);
                    self.decisions.allocator.free(stored.provider_name);
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
    ///
    /// The store always copies the decision key and provider name. `storage`
    ///  controls whether it copies or consumes the artifact data.
    pub fn put_profitable(
        self: *KernelStore,
        decision_key: DecisionKey,
        provider_name: []const u8,
        artifact: Artifact,
        storage: ArtifactStorage,
    ) Allocator.Error!void {
        errdefer if (storage == .take) self.decisions.allocator.free(artifact.data);

        const owned_key = try self.decisions.allocator.dupe(u8, decision_key.bytes);
        errdefer self.decisions.allocator.free(owned_key);
        const owned_name = try self.decisions.allocator.dupe(u8, provider_name);
        errdefer self.decisions.allocator.free(owned_name);
        const owned_data = switch (storage) {
            .copy => try self.decisions.allocator.dupe(u8, artifact.data),
            .take => artifact.data,
        };
        errdefer if (storage == .copy) self.decisions.allocator.free(owned_data);

        try self.decisions.put(owned_key, .{ .profitable = .{
            .provider_name = owned_name,
            .artifact = .{
                .data = owned_data,
                .workspace_bytes = artifact.workspace_bytes,
                .workspace_alignment = artifact.workspace_alignment,
            },
        } });
    }

    /// Record a negative tuning decision (provider declined).
    pub fn put_negative(self: *KernelStore, decision_key: DecisionKey, reason: []const u8) Allocator.Error!void {
        const owned_key = try self.decisions.allocator.dupe(u8, decision_key.bytes);
        errdefer self.decisions.allocator.free(owned_key);
        const owned_reason = try self.decisions.allocator.dupe(u8, reason);
        errdefer self.decisions.allocator.free(owned_reason);
        try self.decisions.put(owned_key, .{ .negative = owned_reason });
    }

    /// Look up a tuning decision. Returns null if the key was never evaluated.
    pub fn get(self: *const KernelStore, decision_key: DecisionKey) ?Decision {
        return self.decisions.get(decision_key.bytes);
    }

    /// Returns true if the key has a profitable decision.
    pub fn is_profitable(self: *const KernelStore, decision_key: DecisionKey) bool {
        const decision = self.decisions.get(decision_key.bytes) orelse return false;
        return switch (decision) {
            .profitable => true,
            .negative => false,
        };
    }
};

/// Runtime entry for one provider.
///
/// Populated before execution begins, typically by `tune()`.
///
/// The execution integration resolves `ProfitableDecision.provider_name` through
///  this table. `dispatch_ctx` carries its type through `TypedPtr`.
pub const DispatchEntry = struct {
    dispatch_fn: DispatchFn,
    prepare_fn: ?PrepareFn = null,
    dispatch_ctx: TypedPtr,
};

/// Maps provider names to dispatch entries used during execution.
///
/// Tuning populates the registry before execution begins. The kernel store
///  contains no function pointers.
pub const DispatchRegistry = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged(DispatchEntry) = .{},

    /// Initialize an empty dispatch registry.
    pub fn init(reg_allocator: std.mem.Allocator) DispatchRegistry {
        return .{ .allocator = reg_allocator };
    }

    /// Release the registry table.
    pub fn deinit(self: *DispatchRegistry) void {
        self.entries.deinit(self.allocator);
    }

    /// Register or replace a provider's dispatch entry.
    ///
    /// `provider_name` is borrowed and must outlive the registry.
    pub fn register(self: *DispatchRegistry, provider_name: []const u8, entry: DispatchEntry) Allocator.Error!void {
        if (self.entries.contains(provider_name)) {
            log.info("dispatch registry replacing provider '{s}' entry", .{provider_name});
        }
        try self.entries.put(self.allocator, provider_name, entry);
    }

    /// Look up a dispatch entry by provider name.
    pub fn get(self: *const DispatchRegistry, provider_name: []const u8) ?DispatchEntry {
        return self.entries.get(provider_name);
    }

    /// Prepare every profitable artifact in `store` for execution.
    ///
    /// Providers without a preparation hook consume their portable artifact
    ///  bytes directly during dispatch. Preparation order is unspecified.
    pub fn prepare(
        self: *const DispatchRegistry,
        store: *const KernelStore,
        ctx: PrepareContext,
    ) PrepareError!void {
        var decisions = store.decisions.iterator();
        while (decisions.next()) |decision| {
            const stored = switch (decision.value_ptr.*) {
                .profitable => |value| value,
                .negative => continue,
            };
            const entry = self.get(stored.provider_name) orelse
                return error.ProviderNotRegistered;
            const prepare_fn = entry.prepare_fn orelse continue;
            try prepare_fn(
                entry.dispatch_ctx,
                stored.artifact.data,
                decision.key_ptr.*,
                ctx,
            );
        }
    }
};

test dot_general_is_matrix_matmul {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{},
        .rhs_batch_dims = &.{},
        .lhs_contracting_dims = &.{1},
        .rhs_contracting_dims = &.{0},
    };
    try std.testing.expect(dot_general_is_matrix_matmul(dg));
}

test "dot_general_is_matrix_matmul rejects non-canonical" {
    {
        const with_batch: pr.DotGeneralParams = .{
            .lhs_batch_dims = &.{0},
            .rhs_batch_dims = &.{0},
            .lhs_contracting_dims = &.{1},
            .rhs_contracting_dims = &.{0},
        };
        try std.testing.expect(!dot_general_is_matrix_matmul(with_batch));
    }
    {
        const wrong_contract: pr.DotGeneralParams = .{
            .lhs_batch_dims = &.{},
            .rhs_batch_dims = &.{},
            .lhs_contracting_dims = &.{0},
            .rhs_contracting_dims = &.{1},
        };
        try std.testing.expect(!dot_general_is_matrix_matmul(wrong_contract));
    }
}

test dot_general_is_canonical_batched_matmul {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{ 0, 1 },
        .rhs_batch_dims = &.{ 0, 1 },
        .lhs_contracting_dims = &.{3},
        .rhs_contracting_dims = &.{2},
    };
    try std.testing.expect(dot_general_is_canonical_batched_matmul(dg, 4, 4));
}

test "dot_general_is_canonical_batched_matmul rejects non-prefix batch" {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{1},
        .rhs_batch_dims = &.{1},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    };
    try std.testing.expect(!dot_general_is_canonical_batched_matmul(dg, 3, 3));
}

test "dot_general_is_canonical_batched_matmul rejects rank mismatch" {
    const dg: pr.DotGeneralParams = .{
        .lhs_batch_dims = &.{0},
        .rhs_batch_dims = &.{0},
        .lhs_contracting_dims = &.{2},
        .rhs_contracting_dims = &.{1},
    };
    try std.testing.expect(!dot_general_is_canonical_batched_matmul(dg, 3, 4));
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
    // A provider without a finalize hook accepts finalization as a no-op.
    provider.finalize();
}

test "kernel store put and get profitable" {
    const testing = std.testing;
    const key = DecisionKey{ .bytes = "kp-test:matmul_f32_128x128" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try store.put_profitable(key, "mirage", .{
        .data = "compiled_kernel_bytes",
        .workspace_bytes = 4096,
        .workspace_alignment = 128,
    }, .copy);

    const decision = store.get(key) orelse return error.TestUnexpectedResult;
    switch (decision) {
        .profitable => |stored| {
            try testing.expectEqualStrings("mirage", stored.provider_name);
            try testing.expectEqualStrings("compiled_kernel_bytes", stored.artifact.data);
            try testing.expectEqual(@as(usize, 4096), stored.artifact.workspace_bytes);
            try testing.expectEqual(@as(usize, 128), stored.artifact.workspace_alignment);
        },
        .negative => return error.TestUnexpectedResult,
    }
    try testing.expect(store.is_profitable(key));
}

test "kernel store takes artifact data without copying" {
    const testing = std.testing;
    const key = DecisionKey{ .bytes = "kp-test:take" };
    const data = try testing.allocator.dupe(u8, "compiled_kernel_bytes");
    const data_ptr = data.ptr;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(key, "test", .{
        .data = data,
    }, .take);

    const decision = store.get(key) orelse return error.TestUnexpectedResult;
    switch (decision) {
        .profitable => |stored| try testing.expectEqual(data_ptr, stored.artifact.data.ptr),
        .negative => return error.TestUnexpectedResult,
    }
}

test "kernel store put and get negative" {
    const testing = std.testing;
    const key = DecisionKey{ .bytes = "kp-test:conv_f32_3x3" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try store.put_negative(key, "unsupported shape");

    const decision = store.get(key) orelse return error.TestUnexpectedResult;
    switch (decision) {
        .profitable => return error.TestUnexpectedResult,
        .negative => |reason| {
            try testing.expectEqualStrings("unsupported shape", reason);
        },
    }
    try testing.expect(!store.is_profitable(key));
}

test "kernel store absent key" {
    const testing = std.testing;
    const key = DecisionKey{ .bytes = "kp-test:nonexistent" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try testing.expect(store.get(key) == null);
    try testing.expect(!store.is_profitable(key));
}

test make_decision_key {
    const testing = std.testing;
    const function_fingerprint = fingerprint.Function{ .bytes = .{0x5a} ** 32 };

    const tvm = try make_decision_key(
        testing.allocator,
        "tvm",
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(tvm.bytes);
    const mirage = try make_decision_key(
        testing.allocator,
        "mirage",
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(mirage.bytes);
    const other_device = try make_decision_key(
        testing.allocator,
        "tvm",
        .{ .platform = .cuda, .ordinal = 1 },
        function_fingerprint,
    );
    defer testing.allocator.free(other_device.bytes);
    const reported_case = try make_decision_key(
        testing.allocator,
        "tvm",
        .{ .platform = .{ .name = "CUDA" } },
        function_fingerprint,
    );
    defer testing.allocator.free(reported_case.bytes);

    try testing.expect(!std.mem.eql(u8, tvm.bytes, mirage.bytes));
    try testing.expect(!std.mem.eql(u8, tvm.bytes, other_device.bytes));
    try testing.expectEqualStrings(tvm.bytes, reported_case.bytes);
}

test "dispatch registry register and get" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var dummy_ctx: u8 = 42;
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = TypedPtr.init(&dummy_ctx),
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u8, 42), entry.dispatch_ctx.cast(u8).*);
    try testing.expect(registry.get("tvm") == null);
}

test "dispatch registry replaces duplicate" {
    const testing = std.testing;

    const Dummy = struct {
        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();

    var ctx1: u8 = 1;
    var ctx2: u8 = 2;
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = TypedPtr.init(&ctx1),
    });
    try registry.register("mirage", .{
        .dispatch_fn = Dummy.dispatch,
        .dispatch_ctx = TypedPtr.init(&ctx2),
    });

    const entry = registry.get("mirage") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u8, 2), entry.dispatch_ctx.cast(u8).*);
}

test "dispatch registry prepares profitable artifacts" {
    const testing = std.testing;

    const State = struct {
        calls: usize = 0,

        fn prepare(
            provider_ctx: TypedPtr,
            artifact_data: []const u8,
            kernel_key: []const u8,
            ctx: PrepareContext,
        ) PrepareError!void {
            const self = provider_ctx.cast(@This());
            if (!std.mem.eql(u8, "compiled_kernel_bytes", artifact_data))
                return error.PrepareFailed;
            if (!std.mem.eql(u8, "kp-test:prepare", kernel_key))
                return error.PrepareFailed;
            if (!ctx.device.platform.eql(.cuda)) return error.UnsupportedDevice;
            self.calls += 1;
        }

        fn dispatch(_: TypedPtr, _: []const u8, _: []const u8, _: DispatchContext) DispatchError!void {}
    };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(.{ .bytes = "kp-test:prepare" }, "test", .{
        .data = "compiled_kernel_bytes",
    }, .copy);
    try store.put_negative(.{ .bytes = "kp-test:negative" }, "unsupported");

    var state: State = .{};
    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();
    try registry.register("test", .{
        .dispatch_fn = State.dispatch,
        .prepare_fn = State.prepare,
        .dispatch_ctx = TypedPtr.init(&state),
    });

    try registry.prepare(&store, .{ .device = .{ .platform = .cuda } });
    try testing.expectEqual(@as(usize, 1), state.calls);
}

test "dispatch registry requires providers for stored artifacts" {
    const testing = std.testing;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put_profitable(.{ .bytes = "kp-test:missing-provider" }, "missing", .{
        .data = "compiled_kernel_bytes",
    }, .copy);

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();
    try testing.expectError(
        error.ProviderNotRegistered,
        registry.prepare(&store, .{ .device = .{ .platform = .cuda } }),
    );
}
