//! TVM compilation orchestration.
//!
//! Combines TIR lowering, target-specific builds, host/device splitting,
//! and linking into shared libraries. No single TVM subsystem owns this —
//! it's our pipeline that ties `tvm/tir/`, `tvm/target/`, and `tvm/runtime/`
//! together.
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const tir = @import("tir.zig");
const runtime = @import("runtime.zig");
const Value = api.Value;
const IRModule = tir.IRModule;
const Target = tir.Target;
const TirPass = tir.TirPass;
const TargetKind = tir.TargetKind;
const RuntimeModule = runtime.RuntimeModule;

const log = std.log.scoped(.@"zg/tvm_compile");

/// Apply the full TIR lowering pipeline and build for a target.
///
/// Pass ordering follows TVM's default_tir_pipeline (tvm/driver/build_module.py).
/// See MEMORY.md "TVM CUDA Pipeline (Critical Pass Order)" for CUDA specifics.
pub fn lower_and_build(allocator: std.mem.Allocator, ir_mod: *IRModule, target: Target, kind: TargetKind) !RuntimeModule {
    // Phase 1: Create composite target with host, then bind.
    // MakePackedAPI requires target->GetHost() to return a valid host target;
    // without it, the function is returned unchanged and buffer_map is not cleared.
    var host_target = switch (kind) {
        .cpu => target,
        .cuda => try Target.create(allocator, .cpu),
    };
    defer if (kind == .cuda) host_target.deinit();

    var composite_target = try target.with_host(allocator, host_target);
    defer composite_target.deinit();

    try ir_mod.apply_pass(allocator, .{ .bind_target = .{ .target = composite_target } });

    // Phase 2: Core lowering
    ir_mod.apply_pass_optional(allocator, .lower_cross_thread_reduction);
    try ir_mod.apply_pass(allocator, .lower_init_block);
    try ir_mod.apply_pass(allocator, .plan_and_update_buffer_allocation);
    try ir_mod.apply_pass(allocator, .convert_blocks_to_opaque);
    ir_mod.apply_pass_optional(allocator, .lift_thread_binding);
    ir_mod.apply_pass_optional(allocator, .{ .compact_buffer_alloc = .{ .is_strict = false } });
    ir_mod.apply_pass_optional(allocator, .lower_match_buffer);
    try ir_mod.apply_pass(allocator, .lower_opaque_block);
    try ir_mod.apply_pass(allocator, .flatten_buffer);

    // Phase 3: Loop transforms
    ir_mod.apply_pass_optional(allocator, .{ .narrow_data_type = .{ .target_bits = 32 } });
    ir_mod.apply_pass_optional(allocator, .loop_partition);
    ir_mod.apply_pass_optional(allocator, .{ .vectorize_loop = .{ .enable = true } });
    ir_mod.apply_pass_optional(allocator, .inject_virtual_thread);
    ir_mod.apply_pass_optional(allocator, .inject_double_buffer);
    ir_mod.apply_pass_optional(allocator, .storage_rewrite);

    try ir_mod.apply_pass(allocator, .simplify);
    ir_mod.apply_pass_optional(allocator, .remove_no_op);
    ir_mod.apply_pass_optional(allocator, .{ .common_subexpr_elim = .{ .enable_cse = true, .enable_equiv = false } });

    // Phase 4: Entry function annotation
    ir_mod.apply_pass_optional(allocator, .verify_memory);
    try ir_mod.apply_pass(allocator, .annotate_entry_func);

    // Phase 5: CUDA-specific pre-SplitHostDevice
    if (kind == .cuda) {
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "shared" } });
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "shared.dyn" } });
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "warp" } });
        ir_mod.apply_pass_optional(allocator, .infer_fragment);
        ir_mod.apply_pass_optional(allocator, .lower_thread_allreduce);
        try ir_mod.apply_pass(allocator, .annotate_device_regions);
    }

    // Phase 6: Host/device split and packed API
    try ir_mod.apply_pass(allocator, .split_host_device);
    if (kind == .cuda) {
        ir_mod.apply_pass_optional(allocator, .merge_shared_memory_allocations);
    }
    try ir_mod.apply_pass(allocator, .make_packed_api);
    ir_mod.apply_pass_optional(allocator, .lower_device_kernel_launch);

    // Phase 7: Target-specific finalization and build
    switch (kind) {
        .cpu => {
            ir_mod.apply_pass_optional(allocator, .lower_tvm_builtin);
            ir_mod.apply_pass_optional(allocator, .lower_custom_datatypes);
            try ir_mod.apply_pass(allocator, .lower_intrin);
            ir_mod.apply_pass_optional(allocator, .lower_device_storage_access_info);
            ir_mod.apply_pass_optional(allocator, .combine_context_call);

            return try build_module(allocator, ir_mod.*, target);
        },
        .cuda => {
            return try build_cuda_module(allocator, ir_mod.*, target);
        },
    }
}

/// Build a (CPU) RuntimeModule from a lowered IRModule via `target.build.llvm`.
fn build_module(allocator: std.mem.Allocator, ir_mod: IRModule, target: Target) !RuntimeModule {
    const result = try api.call_global(allocator, "target.build.llvm", &.{
        ir_mod.as_value(),
        target.as_value(),
    });
    const obj = result.as_object() orelse return error.TvmCallFailed;
    return .{ .handle = .{ .ptr = obj } };
}

/// Build a CUDA RuntimeModule: filter host/device, finalize each, build+link.
fn build_cuda_module(allocator: std.mem.Allocator, ir_mod: IRModule, target: Target) !RuntimeModule {
    // Filter device functions
    var device_mod = try filter_module(allocator, ir_mod, .device);
    defer device_mod.deinit();

    // Device finalization
    device_mod.apply_pass_optional(allocator, .lower_warp_memory);
    device_mod.apply_pass_optional(allocator, .simplify);
    device_mod.apply_pass_optional(allocator, .lower_custom_datatypes);
    device_mod.apply_pass_optional(allocator, .lower_device_storage_access_info);
    device_mod.apply_pass_optional(allocator, .lower_intrin);

    // Build device (CUDA/PTX via NVRTC)
    const device_built = try api.call_global(allocator, "target.build.cuda", &.{ device_mod.as_value(), target.as_value() });
    defer device_built.decref();

    // Filter host functions
    var host_mod = try filter_module(allocator, ir_mod, .host);
    defer host_mod.deinit();

    // Host finalization
    host_mod.apply_pass_optional(allocator, .lower_tvm_builtin);
    host_mod.apply_pass_optional(allocator, .lower_custom_datatypes);
    host_mod.apply_pass_optional(allocator, .lower_intrin);
    host_mod.apply_pass_optional(allocator, .lower_device_storage_access_info);
    host_mod.apply_pass_optional(allocator, .combine_context_call);

    // Build host (LLVM)
    var host_target = try Target.create(allocator, .cpu);
    defer host_target.deinit();
    const host_built = try api.call_global(allocator, "target.build.llvm", &.{ host_mod.as_value(), host_target.as_value() });

    // Link device into host
    _ = try api.call_global(allocator, "ffi.ModuleImportModule", &.{ host_built, device_built });

    const obj = host_built.as_object() orelse return error.TvmCallFailed;
    return .{ .handle = .{ .ptr = obj } };
}

const FilterKind = enum { host, device };

/// Filter an IRModule to keep only host or device functions.
fn filter_module(allocator: std.mem.Allocator, ir_mod: IRModule, kind: FilterKind) !IRModule {
    const filter_name = switch (kind) {
        .host => "tir.transform.FilterHostFunctions",
        .device => "tir.transform.FilterDeviceFunctions",
    };
    // Try the dedicated filter pass first; fall back to manual attribute filtering
    const pass_val = api.call_global(allocator, filter_name, &.{}) catch {
        // Fallback: use a SelectDevice pass or return a copy
        log.warn("filter pass {s} not found, returning unfiltered", .{filter_name});
        ir_mod.handle.incref();
        return .{ .handle = .{ .ptr = ir_mod.handle.ptr }, .type_index = ir_mod.type_index };
    };
    defer pass_val.decref();

    const result = try api.call_global(allocator, "transform.RunPass", &.{ pass_val, ir_mod.as_value() });
    const obj = result.as_object() orelse return error.TvmCallFailed;
    return .{ .handle = .{ .ptr = obj }, .type_index = result.raw.type_index };
}

// ============================================================================
// Linking
// ============================================================================

/// Link .o files into a .so via `zig cc -shared`.
pub fn link_to_shared(allocator: std.mem.Allocator, obj_paths: []const []const u8, so_path: []const u8) !void {
    var argv_list = std.ArrayList([]const u8).empty;
    defer argv_list.deinit(allocator);
    try argv_list.appendSlice(allocator, &.{ "zig", "cc", "-shared", "-fPIC", "-o", so_path });
    try argv_list.appendSlice(allocator, obj_paths);

    var child = std.process.Child.init(argv_list.items, allocator);
    const term = try child.spawnAndWait();

    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                log.err("linker exited with code {d}", .{code});
                return error.TvmCallFailed;
            }
        },
        else => {
            log.err("linker terminated abnormally", .{});
            return error.TvmCallFailed;
        },
    }
    log.debug("linked -> {s}", .{so_path});
}
