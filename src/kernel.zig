//! Kernel-provider selection, artifacts, and dispatch registration.
//!
//! Tuning asks providers to compile callable PR functions and records selections in a
//!  `KernelStore`.
//!
//! Kernelization reads the store, and execution resolves providers through a
//!  separate `DispatchRegistry`.
const std = @import("std");
const artifact_mod = @import("kernel/artifact.zig");
const device = @import("device.zig");
const fingerprint = @import("pr/analysis/fingerprint.zig");
const pr = @import("pr/pr.zig");
const store_mod = @import("kernel/store.zig");
const TypedPtr = @import("utils/rtti.zig").TypedPtr;
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

pub const Artifact = artifact_mod.Artifact;
pub const Candidate = store_mod.Candidate;
pub const SelectionKey = store_mod.SelectionKey;
pub const KernelStore = store_mod.KernelStore;
pub const ProviderCandidate = store_mod.ProviderCandidate;
pub const PutError = store_mod.PutError;
pub const Selection = store_mod.Selection;

/// Failures produced while encoding selection identity.
pub const IdentityError = Allocator.Error || std.Io.Writer.Error;

/// Combine a requested provider, selected device, and function fingerprint.
///
/// Platform names are normalized because `Platform.eql` ignores ASCII case.
/// Caller owns result.
pub fn make_selection_key(
    allocator: Allocator,
    provider_name: []const u8,
    selected_device: device.Device,
    function_fingerprint: fingerprint.Function,
) IdentityError!SelectionKey {
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
pub const DType = @import("dtype.zig").DType;

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
    /// Stable provider name used in selection and dispatch keys.
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

/// Runtime entry for one provider.
///
/// Populated before execution begins, typically by `tune()`.
///
/// The execution integration resolves `ProviderCandidate.provider_name` through
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

    /// Prepare every selected provider artifact in `store` for execution.
    ///
    /// Providers without a preparation hook consume their portable artifact
    ///  bytes directly during dispatch. Preparation order is unspecified.
    pub fn prepare(
        self: *const DispatchRegistry,
        store: *const KernelStore,
        ctx: PrepareContext,
    ) PrepareError!void {
        var selections = store.selections.iterator();
        while (selections.next()) |selection| {
            const stored = switch (selection.value_ptr.candidate) {
                .provider => |value| value,
                .original => continue,
            };
            const entry = self.get(stored.provider_name) orelse
                return error.ProviderNotRegistered;
            const prepare_fn = entry.prepare_fn orelse continue;
            try prepare_fn(
                entry.dispatch_ctx,
                stored.artifact.data,
                selection.key_ptr.*,
                ctx,
            );
        }
    }
};

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

test make_selection_key {
    const testing = std.testing;
    const function_fingerprint = fingerprint.Function{ .bytes = .{0x5a} ** 32 };

    const tvm = try make_selection_key(
        testing.allocator,
        "tvm",
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(tvm.bytes);
    const mirage = try make_selection_key(
        testing.allocator,
        "mirage",
        .{ .platform = .cuda },
        function_fingerprint,
    );
    defer testing.allocator.free(mirage.bytes);
    const other_device = try make_selection_key(
        testing.allocator,
        "tvm",
        .{ .platform = .cuda, .ordinal = 1 },
        function_fingerprint,
    );
    defer testing.allocator.free(other_device.bytes);
    const reported_case = try make_selection_key(
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

test "dispatch registry prepares selected provider artifacts" {
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
    try put_provider_for_test(&store, "kp-test:prepare", "test", "compiled_kernel_bytes");
    try store.put(.{ .bytes = "kp-test:original" }, .{
        .candidate = .original,
        .reason = "unsupported",
    });

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
    try put_provider_for_test(
        &store,
        "kp-test:missing-provider",
        "missing",
        "compiled_kernel_bytes",
    );

    var registry = DispatchRegistry.init(testing.allocator);
    defer registry.deinit();
    try testing.expectError(
        error.ProviderNotRegistered,
        registry.prepare(&store, .{ .device = .{ .platform = .cuda } }),
    );
}

fn put_provider_for_test(
    store: *KernelStore,
    key: []const u8,
    provider_name: []const u8,
    data: []const u8,
) !void {
    try store.put(.{ .bytes = key }, .{
        .candidate = .{ .provider = .{
            .provider_name = provider_name,
            .artifact = .{ .data = try store.allocator().dupe(u8, data) },
        } },
        .reason = "available",
    });
}
