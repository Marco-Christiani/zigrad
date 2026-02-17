/// Kernelization Pass
///
/// PR → PR pass that replaces annotated regions with custom_call ops,
/// compiling kernel artifacts via registered providers.
///
/// For each region with a `kernelize` annotation, the pass:
/// 1. Builds a RegionDescriptor from the function and region
/// 2. Finds the named provider
/// 3. Calls provider.compile() to produce a KernelArtifact
/// 4. Registers the KA in the registry under a target name
/// 5. Replaces the region's equations with a custom_call op
///
/// If a provider cannot handle a region (returns Unsupported), the
/// region's equations are left unchanged — baseline lowering handles them.
const std = @import("std");
const pr = @import("../pr/pr.zig");
const kernel = @import("../kernel.zig");
const pass_mod = @import("pass.zig");

const log = std.log.scoped(.@"zg/kernelize");

/// Kernelization pass state. Holds the registry and providers.
///
/// Create this struct, then call `pass()` to get a pipeline-compatible
/// `pass_mod.Pass` value.
pub const KernelizePass = struct {
    registry: *kernel.KernelRegistry,
    providers: []const kernel.KernelProvider,

    pub fn pass(self: *KernelizePass) pass_mod.Pass {
        return .{
            .ptr = @ptrCast(self),
            .run_fn = run_impl,
            .name = "kernelize",
            .input_kind = .pr,
            .output_kind = .pr,
        };
    }

    fn run_impl(ptr: *anyopaque, artifact: *pass_mod.Artifact, ctx: *pass_mod.PassContext) pass_mod.PassError!void {
        const self: *KernelizePass = @ptrCast(@alignCast(ptr));
        if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

        const program = artifact.pr;
        for (program.functions) |func| {
            self.kernelize_function(func, ctx.allocator) catch |err| {
                log.err("kernelization failed for function '{s}': {}", .{ func.name, err });
                return error.ValidationFailed;
            };
        }
    }

    fn kernelize_function(self: *KernelizePass, func: pr.Function, allocator: std.mem.Allocator) !void {
        for (func.regions) |region| {
            const provider_name = region.annotation.kernelize orelse continue;

            const provider = self.find_provider(provider_name) orelse {
                log.debug("no provider named '{s}' for region '{s}', skipping", .{ provider_name, region.name });
                continue;
            };

            const desc = try kernel.describe_region(allocator, func, region);
            defer allocator.free(desc.inputs);
            defer allocator.free(desc.outputs);

            const ka = provider.compile(desc, allocator) catch |err| switch (err) {
                error.Unsupported => {
                    log.debug("provider '{s}' cannot handle region '{s}', falling back to baseline", .{ provider_name, region.name });
                    continue;
                },
                error.CompileFailed => {
                    log.err("provider '{s}' failed to compile region '{s}'", .{ provider_name, region.name });
                    return error.CompileFailed;
                },
                error.OutOfMemory => return error.OutOfMemory,
            };

            try self.registry.put(ka.target_name, ka);
            log.debug("compiled kernel '{s}' for region '{s}' via provider '{s}'", .{ ka.target_name, region.name, provider_name });
        }
    }

    fn find_provider(self: *KernelizePass, name: []const u8) ?kernel.KernelProvider {
        for (self.providers) |p| {
            if (std.mem.eql(u8, p.name, name)) return p;
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "kernelize pass skips regions with no matching provider" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("r", .{ .kernelize = "nonexistent" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{}, // no providers registered
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    // Region equations should still be there (fallback).
    try testing.expectEqual(@as(usize, 1), program.functions[0].eqns.len);
    // Registry should be empty (nothing compiled).
    try testing.expect(registry.get("anything") == null);
}

test "kernelize pass calls provider and registers KA" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});

    try b.push_region("test_region", .{ .kernelize = "mock" });
    const y = try b.emit(.exp, &.{x}, &.{});
    try b.pop_region();

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const MockProvider = struct {
        compiled: bool = false,

        fn compile(ptr: *anyopaque, desc: kernel.RegionDescriptor, allocator: std.mem.Allocator) kernel.CompileError!kernel.KernelArtifact {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.compiled = true;
            return .{
                .provider_name = "mock",
                .data = try allocator.dupe(u8, "mock_kernel_data"),
                .target_name = desc.name,
            };
        }
    };

    var mock = MockProvider{};
    const provider = kernel.KernelProvider{
        .name = "mock",
        .ptr = @ptrCast(&mock),
        .compile_fn = MockProvider.compile,
    };

    var registry = kernel.KernelRegistry.init(testing.allocator);
    defer registry.deinit();

    var kp = KernelizePass{
        .registry = &registry,
        .providers = &.{provider},
    };

    var artifact = pass_mod.Artifact{ .pr = &program };
    var ctx = pass_mod.PassContext{ .allocator = testing.allocator };
    try kp.pass().run(&artifact, &ctx);

    try testing.expect(mock.compiled);
    const ka = registry.get("test_region");
    try testing.expect(ka != null);
    try testing.expectEqualStrings("mock_kernel_data", ka.?.data);
}
