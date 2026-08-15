//! In-memory kernel selections and their provider artifacts.

const std = @import("std");
const artifact_mod = @import("artifact.zig");
const Artifact = artifact_mod.Artifact;
const Allocator = std.mem.Allocator;

/// Store key for one kernel selection on one device.
pub const SelectionKey = struct {
    bytes: []const u8,
};

/// Provider implementation available for selection.
pub const ProviderCandidate = struct {
    /// Stable name used to resolve the provider during execution.
    provider_name: []const u8,

    /// Artifact produced by the provider.
    artifact: Artifact,
};

/// Implementation eligible to satisfy a kernel request.
pub const Candidate = union(enum) {
    /// Preserve the original callable PR function.
    original,

    /// Replace the call with a provider artifact.
    provider: ProviderCandidate,
};

/// Comparable aggregate timings supporting one selection.
pub const TimingEvidence = struct {
    original_ns: u64,
    selected_ns: u64,
};

/// Selected candidate and the reason it was chosen.
pub const Selection = struct {
    candidate: Candidate,
    reason: []const u8,
    timing: ?TimingEvidence = null,
};

/// Failures produced while storing a selection.
pub const PutError = Allocator.Error || error{
    SelectionExists,
};

/// In-memory selections keyed by request, device, and function fingerprint.
///
/// This is the sole selection source used by kernelization. Persistent
///  provider caches have a separate lifecycle.
pub const KernelStore = struct {
    selections: std.StringHashMap(Selection),

    /// Initialize an empty selection store.
    pub fn init(store_allocator: Allocator) KernelStore {
        return .{ .selections = std.StringHashMap(Selection).init(store_allocator) };
    }

    /// Return the allocator used by this store.
    pub fn allocator(self: *const KernelStore) Allocator {
        return self.selections.allocator;
    }

    /// Release every stored selection allocation.
    pub fn deinit(self: *KernelStore) void {
        var it = self.selections.iterator();
        while (it.next()) |entry| {
            const selection = entry.value_ptr.*;
            switch (selection.candidate) {
                .provider => |stored| {
                    self.selections.allocator.free(stored.artifact.data);
                    self.selections.allocator.free(stored.provider_name);
                },
                .original => {},
            }
            self.selections.allocator.free(selection.reason);
            self.selections.allocator.free(entry.key_ptr.*);
        }
        self.selections.deinit();
        self.* = undefined;
    }

    /// Store the selected candidate for a request.
    pub fn put(
        self: *KernelStore,
        /// Key copied into the store.
        selection_key: SelectionKey,
        /// Selection copied into the store. A provider artifact transfers its
        ///  data allocation to the store, including when insertion fails.
        selection: Selection,
    ) PutError!void {
        errdefer switch (selection.candidate) {
            .original => {},
            .provider => |provider| self.selections.allocator.free(provider.artifact.data),
        };

        if (self.selections.contains(selection_key.bytes)) return error.SelectionExists;

        const owned_key = try self.selections.allocator.dupe(u8, selection_key.bytes);
        errdefer self.selections.allocator.free(owned_key);
        const owned_reason = try self.selections.allocator.dupe(u8, selection.reason);
        errdefer self.selections.allocator.free(owned_reason);
        const owned_candidate: Candidate = switch (selection.candidate) {
            .original => .original,
            .provider => |provider| .{ .provider = .{
                .provider_name = try self.selections.allocator.dupe(u8, provider.provider_name),
                .artifact = provider.artifact,
            } },
        };
        errdefer switch (owned_candidate) {
            .original => {},
            .provider => |provider| self.selections.allocator.free(provider.provider_name),
        };

        try self.selections.put(owned_key, .{
            .candidate = owned_candidate,
            .reason = owned_reason,
            .timing = selection.timing,
        });
    }

    /// Look up a selection. Returns null if the key was never evaluated.
    pub fn get(self: *const KernelStore, selection_key: SelectionKey) ?Selection {
        return self.selections.get(selection_key.bytes);
    }

    /// Return whether the request selects a provider artifact.
    pub fn uses_provider(self: *const KernelStore, selection_key: SelectionKey) bool {
        const selection = self.selections.get(selection_key.bytes) orelse return false;
        return switch (selection.candidate) {
            .provider => true,
            .original => false,
        };
    }
};

test "kernel store selects a provider artifact" {
    const testing = std.testing;
    const key = SelectionKey{ .bytes = "kp-test:matmul_f32_128x128" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    const data = try testing.allocator.dupe(u8, "compiled_kernel_bytes");
    try store.put(key, .{
        .candidate = .{ .provider = .{
            .provider_name = "mirage",
            .artifact = .{
                .data = data,
                .workspace_bytes = 4096,
                .workspace_alignment = 128,
            },
        } },
        .reason = "available",
        .timing = .{ .original_ns = 120, .selected_ns = 90 },
    });

    const selection = store.get(key) orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .provider => |stored| {
            try testing.expectEqualStrings("mirage", stored.provider_name);
            try testing.expectEqualStrings("compiled_kernel_bytes", stored.artifact.data);
            try testing.expectEqual(@as(usize, 4096), stored.artifact.workspace_bytes);
            try testing.expectEqual(@as(usize, 128), stored.artifact.workspace_alignment);
        },
        .original => return error.TestUnexpectedResult,
    }
    try std.testing.expectEqual(@as(u64, 120), selection.timing.?.original_ns);
    try std.testing.expectEqual(@as(u64, 90), selection.timing.?.selected_ns);
    try testing.expect(store.uses_provider(key));
}

test "kernel store takes artifact data without copying" {
    const testing = std.testing;
    const key = SelectionKey{ .bytes = "kp-test:take" };
    const data = try testing.allocator.dupe(u8, "compiled_kernel_bytes");
    const data_ptr = data.ptr;

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put(key, .{
        .candidate = .{ .provider = .{
            .provider_name = "test",
            .artifact = .{ .data = data },
        } },
        .reason = "available",
    });

    const selection = store.get(key) orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .provider => |stored| try testing.expectEqual(data_ptr, stored.artifact.data.ptr),
        .original => return error.TestUnexpectedResult,
    }
}

test "kernel store selects the original candidate" {
    const testing = std.testing;
    const key = SelectionKey{ .bytes = "kp-test:conv_f32_3x3" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();
    try store.put(key, .{
        .candidate = .original,
        .reason = "provider does not support the shape",
    });

    const selection = store.get(key) orelse return error.TestUnexpectedResult;
    switch (selection.candidate) {
        .provider => return error.TestUnexpectedResult,
        .original => try testing.expectEqualStrings(
            "provider does not support the shape",
            selection.reason,
        ),
    }
    try testing.expect(!store.uses_provider(key));
}

test "kernel store has no implicit selection" {
    const testing = std.testing;
    const key = SelectionKey{ .bytes = "kp-test:nonexistent" };

    var store = KernelStore.init(testing.allocator);
    defer store.deinit();

    try testing.expect(store.get(key) == null);
    try testing.expect(!store.uses_provider(key));
}
