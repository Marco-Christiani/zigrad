const std = @import("std");

const pr = @import("../pr/pr.zig");
const pipeline_spec = @import("../pipeline_spec.zig");

const im_stablehlo = @import("../im/stablehlo/im.zig");
const xla_toolchain = @import("../toolchain/xla/compile.zig");
const pjrt_runtime = @import("../runtime/pjrt/runtime.zig");
const pjrt_types = @import("../ffi/pjrt/types.zig");

pub const ExecutableArtifact = union(enum) {
    /// A runtime-bound executable handle (JIT).
    loaded: pjrt_types.LoadedExecutable,

    /// A platform/version-specific cache artifact (still JIT; not “true AOT”).
    serialized: []u8,

    pub fn deinit(self: *ExecutableArtifact, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .loaded => |*exe| exe.deinit(),
            .serialized => |bytes| allocator.free(bytes),
        }
    }
};

pub const Pipeline = struct {
    allocator: std.mem.Allocator,
    spec: pipeline_spec.PipelineSpec,
    pjrt: ?pjrt_runtime.Runtime = null,

    pub fn init(allocator: std.mem.Allocator, spec_in: pipeline_spec.PipelineSpec) !Pipeline {
        try spec_in.validate();

        var out = Pipeline{
            .allocator = allocator,
            .spec = spec_in,
            .pjrt = null,
        };
        errdefer out.deinit();

        switch (spec_in.runtime) {
            .pjrt => |cfg| {
                // Take ownership of plugin_path so `PipelineSpec` and this Pipeline can be self-contained.
                const owned_path = try allocator.dupe(u8, cfg.plugin_path);
                out.spec.runtime = .{ .pjrt = .{ .plugin_path = owned_path } };

                out.pjrt = try pjrt_runtime.Runtime.init(allocator, owned_path);
            },
            else => return error.UnsupportedPipeline,
        }

        return out;
    }

    pub fn deinit(self: *Pipeline) void {
        if (self.pjrt) |*rt| rt.deinit();
        self.pjrt = null;

        // Free spec-owned strings.
        switch (self.spec.runtime) {
            .pjrt => |cfg| self.allocator.free(cfg.plugin_path),
            else => {},
        }
    }

    pub fn runtimePjrt(self: *Pipeline) !*pjrt_runtime.Runtime {
        return &(self.pjrt orelse return error.MissingRuntime);
    }

    pub fn devices(self: *Pipeline, allocator: std.mem.Allocator) ![]pjrt_types.Device {
        return (try self.runtimePjrt()).devices(allocator);
    }

    fn realizeIm(self: *const Pipeline, allocator: std.mem.Allocator, func: pr.Function) !im_stablehlo.IM {
        const emit_text = self.spec.im_profile == .stablehlo_mlir_text;
        return im_stablehlo.realize(allocator, func, .{ .emit_text = emit_text });
    }

    pub fn compilePr(
        self: *Pipeline,
        func: pr.Function,
        device: *const pjrt_types.Device,
        options: xla_toolchain.CompileOptions,
    ) !ExecutableArtifact {
        try self.spec.validate();

        if (self.spec.primary.kind() != .xla or self.spec.runtime.kind() != .pjrt)
            return error.UnsupportedPipeline;

        const rt = try self.runtimePjrt();

        // For now, compilation for xla+pjrt is inherently runtime-hosted.
        if (self.spec.mode == .aot) return error.UnsupportedPipeline;

        var im = try self.realizeIm(self.allocator, func);
        defer im.deinit();

        switch (self.spec.mode) {
            .jit => {
                const exe = try xla_toolchain.compile(self.allocator, rt.getClient(), device, im, options);
                return .{ .loaded = exe };
            },
            .jit_cache => {
                const bytes = try xla_toolchain.compileSerialized(self.allocator, rt.getClient(), device, im, options);
                return .{ .serialized = bytes };
            },
            .aot => return error.UnsupportedPipeline,
        }
    }

    pub fn loadSerializedExecutable(
        self: *Pipeline,
        serialized_executable: []const u8,
        overridden_compile_options: ?[]const u8,
    ) !pjrt_types.LoadedExecutable {
        try self.spec.validate();

        if (self.spec.runtime.kind() != .pjrt) return error.UnsupportedPipeline;
        const rt = try self.runtimePjrt();
        return rt.loadSerializedExecutable(serialized_executable, overridden_compile_options);
    }
};
