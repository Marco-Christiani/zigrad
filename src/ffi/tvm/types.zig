//! Typed TVM object wrappers.
//!
//! Each struct wraps an ObjectHandle with domain-specific methods.
//! All TVM C types are contained here — callers in src/tvm/ see only
//! Zig types. Mirrors src/ffi/pjrt/types.zig in role.
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const dlpack = @import("../dlpack.zig");
const Value = api.Value;
const ObjectHandle = api.ObjectHandle;
const TvmError = api.TvmError;

/// Target kind for TVM compilation.
pub const TargetKind = enum { cpu, cuda };

const log = std.log.scoped(.@"zg/tvm_types");

// ============================================================================
// Array — TVM runtime Array wrapper
// ============================================================================

/// Typed wrapper for TVM's `ffi.Array`.
///
/// Provides typed access to array construction, length, and element access.
/// Owns the underlying TVM object handle and decrements its refcount on deinit.
pub const Array = struct {
    handle: ObjectHandle,
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    /// Wrap an existing TVM array Value, taking a reference.
    ///
    /// Increments the refcount so `deinit` is safe and symmetric.
    /// Use this for callback arguments where the caller retains ownership.
    pub fn wrap(val: Value) !Array {
        const obj = val.as_object() orelse return error.TvmCallFailed;
        _ = c.TVMFFIObjectIncRef(obj);
        return .{ .handle = .{ .ptr = obj }, .type_index = val.raw.type_index };
    }

    /// Construct a TVM Array from a slice of Values.
    pub fn from_values(allocator: std.mem.Allocator, items: []const Value) !Array {
        const result = try api.call_global(allocator, "ffi.Array", items);
        return .{
            .handle = .{ .ptr = result.as_object() orelse return error.TvmCallFailed },
            .type_index = result.raw.type_index,
        };
    }

    /// Number of elements in the array.
    pub fn len(self: Array, allocator: std.mem.Allocator) !usize {
        const result = try api.call_global(allocator, "ffi.ArraySize", &.{self.as_value()});
        return @intCast(result.as_int() orelse return error.TvmCallFailed);
    }

    /// Get the element at `idx`.
    pub fn get(self: Array, allocator: std.mem.Allocator, idx: usize) !Value {
        return api.call_global(allocator, "ffi.ArrayGetItem", &.{
            self.as_value(), Value.int(@intCast(idx)),
        });
    }

    pub fn as_value(self: Array) Value {
        return self.handle.to_value(self.type_index);
    }

    pub fn deinit(self: *Array) void {
        self.handle.deinit();
    }
};

// ============================================================================
// IRModule
// ============================================================================

pub const IRModule = struct {
    handle: ObjectHandle,
    /// TVM runtime type index, preserved from the FFI call that created this object.
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    pub fn deinit(self: *IRModule) void {
        self.handle.deinit();
    }

    pub fn as_value(self: IRModule) Value {
        return self.handle.to_value(self.type_index);
    }

    /// Apply a single TIR transform pass to this module (in-place replacement).
    pub fn apply_pass(self: *IRModule, allocator: std.mem.Allocator, pass: TirPass) !void {
        // Construct the pass object
        const pass_val = try pass.create(allocator);
        defer {
            if (pass_val.as_object()) |obj| _ = c.TVMFFIObjectDecRef(obj);
        }

        const old_ptr = self.handle.ptr;

        // Run it: transform.RunPass(pass, module) -> module
        const result = try api.call_global(allocator, "transform.RunPass", &.{ pass_val, self.as_value() });

        const new_obj = result.as_object() orelse return error.TvmCallFailed;
        // Replace handle (decref old if different)
        if (self.handle.ptr != new_obj) {
            self.handle.deinit();
        }
        self.handle.ptr = new_obj;
        self.type_index = result.raw.type_index;
        log.debug("applied {s} (ptr {s}, type_index {d}→{d})", .{
            pass.name(),
            if (old_ptr != new_obj) "changed" else "same",
            @as(c_int, if (old_ptr == new_obj) self.type_index else 0),
            result.raw.type_index,
        });
    }

    /// Apply a pass, ignoring failure (for optional/non-fatal passes).
    pub fn apply_pass_optional(self: *IRModule, allocator: std.mem.Allocator, pass: TirPass) void {
        self.apply_pass(allocator, pass) catch {
            log.debug("optional pass {s} failed (non-fatal)", .{pass.name()});
        };
    }
};

// ============================================================================
// Target
// ============================================================================

pub const Target = struct {
    handle: ObjectHandle,
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    pub fn deinit(self: *Target) void {
        self.handle.deinit();
    }

    pub fn as_value(self: Target) Value {
        return self.handle.to_value(self.type_index);
    }

    /// Create a Target from a TargetKind.
    ///
    /// For CPU targets, includes `-num-cores` (required by MetaSchedule).
    pub fn create(allocator: std.mem.Allocator, kind: TargetKind) !Target {
        const s = switch (kind) {
            .cpu => blk: {
                const ncores = std.Thread.getCpuCount() catch 1;
                break :blk try std.fmt.allocPrintSentinel(allocator, "llvm -num-cores {d}", .{ncores}, 0);
            },
            .cuda => try std.fmt.allocPrintSentinel(allocator, "nvidia/nvidia-a100", .{}, 0),
        };
        defer allocator.free(s);

        const result = try api.call_global(allocator, "target.Target", &.{Value.str(s)});
        const obj = result.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj }, .type_index = result.raw.type_index };
    }

    /// Create a composite target with a host target attached.
    pub fn with_host(self: Target, allocator: std.mem.Allocator, host: Target) !Target {
        const result = try api.call_global(allocator, "target.WithHost", &.{
            self.as_value(),
            host.as_value(),
        });
        const obj = result.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj }, .type_index = result.raw.type_index };
    }
};

// ============================================================================
// RuntimeModule
// ============================================================================

pub const RuntimeModule = struct {
    handle: ObjectHandle,

    pub fn deinit(self: *RuntimeModule) void {
        self.handle.deinit();
    }

    fn as_value(self: RuntimeModule) Value {
        return self.handle.to_value(c.kTVMFFIModule);
    }

    /// Load a compiled module (.so) from disk.
    pub fn load_from_file(allocator: std.mem.Allocator, path: []const u8) !RuntimeModule {
        const path_z = try api.cstr_alloc(allocator, path);
        defer allocator.free(path_z);
        const result = try api.call_global(allocator, "ffi.ModuleLoadFromFile", &.{
            Value.str(path_z),
        });
        return .{ .handle = .{ .ptr = result.as_object() orelse return error.TvmCallFailed } };
    }

    /// Get a packed function from this module by name.
    pub fn get_function(self: RuntimeModule, allocator: std.mem.Allocator, name: []const u8, query_imports: bool) !Value {
        const name_z = try api.cstr_alloc(allocator, name);
        defer allocator.free(name_z);
        return api.call_global(allocator, "ffi.ModuleGetFunction", &.{
            self.as_value(), Value.str(name_z), Value.boolean(query_imports),
        });
    }

    /// Write the module to a file in the given format ("o", "so", "ptx", etc.).
    pub fn write_to_file(self: RuntimeModule, allocator: std.mem.Allocator, path: []const u8, format: []const u8) !void {
        const path_z = try api.cstr_alloc(allocator, path);
        defer allocator.free(path_z);
        const fmt_z = try api.cstr_alloc(allocator, format);
        defer allocator.free(fmt_z);

        _ = try api.call_global(allocator, "ffi.ModuleWriteToFile", &.{
            self.as_value(),
            Value.str(path_z),
            Value.str(fmt_z),
        });
        log.debug("wrote module to {s} (format={s})", .{ path, format });
    }

    /// Pack device module imports into an LLVM blob (for CUDA .so linking).
    pub fn pack_imports_to_llvm(self: RuntimeModule, allocator: std.mem.Allocator) !RuntimeModule {
        const result = try api.call_global(allocator, "runtime.ModulePackImportsToLLVM", &.{
            self.as_value(),
            Value.boolean(false), // system_lib
            Value.str("llvm"),
            Value.str(""),
        });
        const obj = result.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj } };
    }

    /// Export the module to a shared library (.so).
    ///
    /// For CPU: writes a single .o and links to .so.
    /// For CUDA: writes host .o + device .o (packed LLVM blob), then links both.
    pub fn export_shared(self: RuntimeModule, allocator: std.mem.Allocator, so_path: []const u8, kind: TargetKind) !void {
        const obj_path = try std.fmt.allocPrint(allocator, "{s}.host.o", .{so_path});
        defer allocator.free(obj_path);

        try self.write_to_file(allocator, obj_path, "o");

        switch (kind) {
            .cpu => {
                try link_to_shared(allocator, &.{obj_path}, so_path);
            },
            .cuda => {
                const devc_obj_path = try std.fmt.allocPrint(allocator, "{s}.devc.o", .{so_path});
                defer allocator.free(devc_obj_path);

                var pack_mod = try self.pack_imports_to_llvm(allocator);
                defer pack_mod.deinit();
                try pack_mod.write_to_file(allocator, devc_obj_path, "o");

                try link_to_shared(allocator, &.{ obj_path, devc_obj_path }, so_path);
            },
        }
        log.info("exported {s}", .{so_path});
    }
};

// ============================================================================
// TirPass — compile-time checked TIR pass names
// ============================================================================

/// TIR transform passes as a tagged union. Compile-time checked names
/// prevent string typos. Passes with arguments carry their args inline.
pub const TirPass = union(enum) {
    // No-arg passes
    lower_cross_thread_reduction,
    lower_init_block,
    plan_and_update_buffer_allocation,
    convert_blocks_to_opaque,
    lift_thread_binding,
    lower_match_buffer,
    lower_opaque_block,
    flatten_buffer,
    loop_partition,
    inject_virtual_thread,
    inject_double_buffer,
    storage_rewrite,
    simplify,
    remove_no_op,
    verify_memory,
    annotate_entry_func,
    infer_fragment,
    lower_thread_allreduce,
    annotate_device_regions,
    split_host_device,
    merge_shared_memory_allocations,
    make_packed_api,
    lower_device_kernel_launch,
    lower_tvm_builtin,
    lower_custom_datatypes,
    lower_intrin,
    lower_device_storage_access_info,
    combine_context_call,
    lower_warp_memory,

    // Passes with arguments
    bind_target: struct { target: Target },
    thread_sync: struct { scope: []const u8 },
    compact_buffer_alloc: struct { is_strict: bool },
    narrow_data_type: struct { target_bits: i64 },
    vectorize_loop: struct { enable: bool },
    common_subexpr_elim: struct { enable_cse: bool, enable_equiv: bool },

    /// Returns the TVM global function name for this pass.
    pub fn name(self: TirPass) []const u8 {
        return switch (self) {
            .lower_cross_thread_reduction => "tir.transform.LowerCrossThreadReduction",
            .lower_init_block => "tir.transform.LowerInitBlock",
            .plan_and_update_buffer_allocation => "tir.transform.PlanAndUpdateBufferAllocationLocation",
            .convert_blocks_to_opaque => "tir.transform.ConvertBlocksToOpaque",
            .lift_thread_binding => "tir.transform.LiftThreadBinding",
            .lower_match_buffer => "tir.transform.LowerMatchBuffer",
            .lower_opaque_block => "tir.transform.LowerOpaqueBlock",
            .flatten_buffer => "tir.transform.FlattenBuffer",
            .loop_partition => "tir.transform.LoopPartition",
            .inject_virtual_thread => "tir.transform.InjectVirtualThread",
            .inject_double_buffer => "tir.transform.InjectDoubleBuffer",
            .storage_rewrite => "tir.transform.StorageRewrite",
            .simplify => "tir.transform.Simplify",
            .remove_no_op => "tir.transform.RemoveNoOp",
            .verify_memory => "tir.transform.VerifyMemory",
            .annotate_entry_func => "tir.transform.AnnotateEntryFunc",
            .infer_fragment => "tir.transform.InferFragment",
            .lower_thread_allreduce => "tir.transform.LowerThreadAllreduce",
            .annotate_device_regions => "tir.transform.AnnotateDeviceRegions",
            .split_host_device => "tir.transform.SplitHostDevice",
            .merge_shared_memory_allocations => "tir.transform.MergeSharedMemoryAllocations",
            .make_packed_api => "tir.transform.MakePackedAPI",
            .lower_device_kernel_launch => "tir.transform.LowerDeviceKernelLaunch",
            .lower_tvm_builtin => "tir.transform.LowerTVMBuiltin",
            .lower_custom_datatypes => "tir.transform.LowerCustomDatatypes",
            .lower_intrin => "tir.transform.LowerIntrin",
            .lower_device_storage_access_info => "tir.transform.LowerDeviceStorageAccessInfo",
            .combine_context_call => "tir.transform.CombineContextCall",
            .lower_warp_memory => "tir.transform.LowerWarpMemory",
            .bind_target => "tir.transform.BindTarget",
            .thread_sync => "tir.transform.ThreadSync",
            .compact_buffer_alloc => "tir.transform.CompactBufferAllocation",
            .narrow_data_type => "tir.transform.NarrowDataType",
            .vectorize_loop => "tir.transform.VectorizeLoop",
            .common_subexpr_elim => "tir.transform.CommonSubexprElimTIR",
        };
    }

    /// Create the TVM pass object by calling the global function with args.
    fn create(self: TirPass, allocator: std.mem.Allocator) TvmError!Value {
        return switch (self) {
            // No-arg passes
            .lower_cross_thread_reduction,
            .lower_init_block,
            .plan_and_update_buffer_allocation,
            .convert_blocks_to_opaque,
            .lift_thread_binding,
            .lower_match_buffer,
            .lower_opaque_block,
            .flatten_buffer,
            .loop_partition,
            .inject_virtual_thread,
            .inject_double_buffer,
            .storage_rewrite,
            .simplify,
            .remove_no_op,
            .verify_memory,
            .annotate_entry_func,
            .infer_fragment,
            .lower_thread_allreduce,
            .annotate_device_regions,
            .split_host_device,
            .merge_shared_memory_allocations,
            .make_packed_api,
            .lower_device_kernel_launch,
            .lower_tvm_builtin,
            .lower_custom_datatypes,
            .lower_intrin,
            .lower_device_storage_access_info,
            .combine_context_call,
            .lower_warp_memory,
            => try api.call_global(allocator, self.name(), &.{}),

            // Passes with arguments
            .bind_target => |args| try api.call_global(allocator, self.name(), &.{args.target.as_value()}),
            .thread_sync => |args| blk: {
                const s = try api.cstr_alloc(allocator, args.scope);
                defer allocator.free(s);
                break :blk try api.call_global(allocator, self.name(), &.{Value.str(s)});
            },
            .compact_buffer_alloc => |args| try api.call_global(allocator, self.name(), &.{Value.boolean(args.is_strict)}),
            .narrow_data_type => |args| try api.call_global(allocator, self.name(), &.{Value.int(args.target_bits)}),
            .vectorize_loop => |args| try api.call_global(allocator, self.name(), &.{Value.boolean(args.enable)}),
            .common_subexpr_elim => |args| try api.call_global(allocator, self.name(), &.{
                Value.boolean(args.enable_cse),
                Value.boolean(args.enable_equiv),
            }),
        };
    }
};

// ============================================================================
// Build matmul TIR module
// ============================================================================

/// Build a matmul IRModule from shapes via TE (topi.matmul).
///
/// Creates A[M,K] @ B[K,N] = C[M,N] via te.Placeholder + topi.matmul,
/// wraps in CreatePrimFunc + IRModule with global_symbol="main".
pub fn build_matmul_tir(allocator: std.mem.Allocator, m: usize, n: usize, k: usize) !IRModule {
    const m_i: i64 = @intCast(m);
    const n_i: i64 = @intCast(n);
    const k_i: i64 = @intCast(k);

    // Shapes
    const shape_a = try api.call_global(allocator, "ffi.Array", &.{ Value.int(m_i), Value.int(k_i) });
    defer shape_a.decref();
    const shape_b = try api.call_global(allocator, "ffi.Array", &.{ Value.int(k_i), Value.int(n_i) });
    defer shape_b.decref();

    // Placeholder tensors
    const dtype_z = try api.cstr_alloc(allocator, "float32");
    defer allocator.free(dtype_z);
    const name_a_z = try api.cstr_alloc(allocator, "A");
    defer allocator.free(name_a_z);
    const name_b_z = try api.cstr_alloc(allocator, "B");
    defer allocator.free(name_b_z);

    const tensor_a = try api.call_global(allocator, "te.Placeholder", &.{
        shape_a, Value.str(dtype_z), Value.str(name_a_z),
    });
    const tensor_b = try api.call_global(allocator, "te.Placeholder", &.{
        shape_b, Value.str(dtype_z), Value.str(name_b_z),
    });
    log.debug("created placeholders A[{d},{d}] B[{d},{d}]", .{ m, k, k, n });

    // Matmul via topi
    const tensor_c = api.call_global(allocator, "topi.matmul", &.{
        tensor_a, tensor_b, Value.boolean(false), Value.boolean(false),
    }) catch |err1| blk: {
        log.warn("topi.matmul failed ({s}), trying topi.nn.matmul", .{@errorName(err1)});
        break :blk try api.call_global(allocator, "topi.nn.matmul", &.{ tensor_a, tensor_b });
    };
    log.info("created matmul C[{d},{d}] = A[{d},{d}] @ B[{d},{d}]", .{ m, n, m, k, k, n });

    // CreatePrimFunc
    const tensors_arr = try api.call_global(allocator, "ffi.Array", &.{ tensor_a, tensor_b, tensor_c });
    const prim_func = try api.call_global(allocator, "te.CreatePrimFunc", &.{ tensors_arr, Value.none() });

    // Attach global_symbol="main"
    const gs_z = try api.cstr_alloc(allocator, "global_symbol");
    defer allocator.free(gs_z);
    const main_z = try api.cstr_alloc(allocator, "main");
    defer allocator.free(main_z);

    const prim_func_attr = try api.call_global(allocator, "ir.BaseFuncWithAttr", &.{
        prim_func, Value.str(gs_z), Value.str(main_z),
    });
    prim_func.decref();

    // Wrap in IRModule
    const main_z2 = try api.cstr_alloc(allocator, "main");
    defer allocator.free(main_z2);

    const global_var = try api.call_global(allocator, "ir.GlobalVar", &.{Value.str(main_z2)});
    const func_map = try api.call_global(allocator, "ffi.Map", &.{ global_var, prim_func_attr });
    const empty_map = try api.call_global(allocator, "ffi.Map", &.{});
    const ir_mod = try api.call_global(allocator, "ir.IRModule", &.{ func_map, Value.none(), empty_map });

    const obj = ir_mod.as_object() orelse return error.TvmCallFailed;
    log.info("created matmul IRModule ({d}x{d}x{d})", .{ m, n, k });
    return .{ .handle = .{ .ptr = obj }, .type_index = ir_mod.raw.type_index };
}

// ============================================================================
// TIR lowering pipeline
// ============================================================================

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
// Tensor — DLPack ↔ TVM tensor bridge
// ============================================================================

pub const Tensor = struct {
    handle: ObjectHandle,

    pub fn deinit(self: *Tensor) void {
        self.handle.deinit();
    }

    /// Create a TVM tensor from a DLPack ManagedTensor.
    pub fn from_dlpack(managed: *dlpack.ManagedTensor) TvmError!Tensor {
        var out: c.TVMFFIObjectHandle = null;
        if (c.TVMFFITensorFromDLPack(@ptrCast(managed), 0, 0, &out) != 0 or out == null) {
            return error.TvmCallFailed;
        }
        return .{ .handle = .{ .ptr = out } };
    }

    /// Allocate a TVM tensor on the given device and copy data from host.
    ///
    /// `shape` must remain valid for the lifetime of the returned tensor
    /// (TVM stores the shape pointer internally).
    pub fn allocate(allocator: std.mem.Allocator, data: []f32, shape: []i64, device_type: dlpack.DeviceType) TvmError!Tensor {
        // 1. Create Shape object
        var shape_vals: [4]Value = undefined;
        for (shape, 0..) |dim, i| {
            shape_vals[i] = Value.int(dim);
        }
        const shape_obj = try api.call_global(allocator, "ffi.Shape", shape_vals[0..shape.len]);
        defer shape_obj.decref();

        // 2. Allocate empty tensor on device
        const tensor_val = try api.call_global(allocator, "runtime.TVMTensorAllocWithScope", &.{
            shape_obj,
            dtype_value(.float, 32),
            device_value(device_type, 0),
            Value.none(),
        });

        // 3. Copy host data to device tensor
        const nbytes = data.len * @sizeOf(f32);
        _ = try api.call_global(allocator, "runtime.TVMTensorCopyFromBytes", &.{
            tensor_val,
            ptr_value(@ptrCast(@constCast(data.ptr))),
            Value.int(@intCast(nbytes)),
        });

        const obj = tensor_val.as_object() orelse return error.TvmCallFailed;
        return .{ .handle = .{ .ptr = obj } };
    }

    /// Copy tensor data back to host memory.
    pub fn copy_to_host(self: Tensor, allocator: std.mem.Allocator, dest: []f32) TvmError!void {
        const nbytes = dest.len * @sizeOf(f32);
        _ = try api.call_global(allocator, "runtime.TVMTensorCopyToBytes", &.{
            self.as_value(),
            ptr_value(@ptrCast(dest.ptr)),
            Value.int(@intCast(nbytes)),
        });
    }

    /// Wrap as a Value for passing to TVM function calls.
    pub fn as_value(self: Tensor) Value {
        return Value.from_object(self.handle.ptr, c.kTVMFFITensor);
    }
};

// Private helpers for constructing special Value types needed by Tensor methods.

fn device_value(device_type: dlpack.DeviceType, device_id: i32) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDevice;
    v.unnamed_1.v_device = .{ .device_type = @intCast(@intFromEnum(device_type)), .device_id = device_id };
    return .{ .raw = v };
}

fn dtype_value(code: dlpack.DataTypeCode, bits: u8) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIDataType;
    v.unnamed_1.v_dtype = .{ .code = @intCast(@intFromEnum(code)), .bits = bits, .lanes = 1 };
    return .{ .raw = v };
}

fn ptr_value(ptr: *anyopaque) Value {
    var v = std.mem.zeroes(c.TVMFFIAny);
    v.type_index = c.kTVMFFIOpaquePtr;
    v.unnamed_1.v_int64 = @bitCast(@intFromPtr(ptr));
    return .{ .raw = v };
}

// ============================================================================
// MetaSchedule — typed constructors for TVM auto-tuning objects
// ============================================================================

/// MetaSchedule typed constructors.
///
/// Each function wraps one or more `call_global` calls to construct TVM
/// MetaSchedule objects with compile-time checked names. These are
/// constructor-only wrappers — they return Values that the caller manages.
pub const MetaSchedule = struct {
    /// Create schedule rules for the given target kind.
    pub fn schedule_rules(allocator: std.mem.Allocator, kind: TargetKind) !Value {
        const name = switch (kind) {
            .cpu => "meta_schedule.ScheduleRuleDefaultLLVM",
            .cuda => "meta_schedule.ScheduleRuleDefaultCUDA",
        };
        return api.call_global(allocator, name, &.{});
    }

    /// Create a SpaceGenerator with post-order apply.
    pub fn space_generator(allocator: std.mem.Allocator, rules: Value) !Value {
        return api.call_global(allocator, "meta_schedule.SpaceGeneratorPostOrderApply", &.{
            Value.none(), // f_block_filter
            rules,
            Value.none(), // postprocs
            Value.none(), // mutator_probs
        });
    }

    pub const SearchStrategyOpts = struct {
        population_size: i64 = 512,
        init_measured_ratio: f64 = 0.2,
        init_min_unmeasured: i64 = 50,
        max_fail_count: i64 = 5,
        genetic_num_iters: i64 = 3,
        genetic_mutate_prob: f64 = 0.85,
        genetic_max_fail_count: i64 = 10,
        eps_greedy: f64 = 0.05,
    };

    /// Create an evolutionary search strategy.
    pub fn search_strategy(allocator: std.mem.Allocator, opts: SearchStrategyOpts) !Value {
        return api.call_global(allocator, "meta_schedule.SearchStrategyEvolutionarySearch", &.{
            Value.int(opts.population_size),
            Value.float(opts.init_measured_ratio),
            Value.int(opts.init_min_unmeasured),
            Value.int(opts.max_fail_count),
            Value.int(opts.genetic_num_iters),
            Value.float(opts.genetic_mutate_prob),
            Value.int(opts.genetic_max_fail_count),
            Value.float(opts.eps_greedy),
        });
    }

    /// Create a JSON database for persisting tuning records.
    pub fn json_database(
        allocator: std.mem.Allocator,
        workload_path: [:0]const u8,
        record_path: [:0]const u8,
    ) !Value {
        const structural_z = try api.cstr_alloc(allocator, "structural");
        defer allocator.free(structural_z);
        return api.call_global(allocator, "meta_schedule.DatabaseJSONDatabase", &.{
            Value.str(workload_path),
            Value.str(record_path),
            Value.boolean(true), // allow_missing
            Value.str(structural_z),
        });
    }

    pub const TuneContextOpts = struct {
        ir_mod: Value,
        target: Value,
        space_gen: Value,
        search_strat: Value,
        task_name: [:0]const u8,
        num_threads: i64 = 1,
        rand_state: i64 = 42,
        logger: Value,
    };

    /// Create a TuneContext.
    pub fn tune_context(allocator: std.mem.Allocator, opts: TuneContextOpts) !Value {
        return api.call_global(allocator, "meta_schedule.TuneContext", &.{
            opts.ir_mod,
            opts.target,
            opts.space_gen,
            opts.search_strat,
            Value.str(opts.task_name),
            Value.int(opts.num_threads),
            Value.int(opts.rand_state),
            opts.logger,
        });
    }

    /// Create a PyBuilder wrapping a packed function callback.
    pub fn py_builder(allocator: std.mem.Allocator, func: Value) !Value {
        return api.call_global(allocator, "meta_schedule.BuilderPyBuilder", &.{func});
    }

    /// Create a PyRunner wrapping a packed function callback.
    pub fn py_runner(allocator: std.mem.Allocator, func: Value) !Value {
        return api.call_global(allocator, "meta_schedule.RunnerPyRunner", &.{func});
    }

    /// Create a PyCostModel with load/save/update/predict/as_string callbacks.
    pub fn py_cost_model(
        allocator: std.mem.Allocator,
        f_load: Value,
        f_save: Value,
        f_update: Value,
        f_predict: Value,
        f_as_string: Value,
    ) !Value {
        return api.call_global(allocator, "meta_schedule.CostModelPyCostModel", &.{
            f_load, f_save, f_update, f_predict, f_as_string,
        });
    }

    pub const TaskSchedulerOpts = struct {
        logger: Value,
        alpha: f64 = 0.8,
        window_size: i64 = 3,
        seed: i64 = 42,
    };

    /// Create a gradient-based task scheduler.
    pub fn task_scheduler(allocator: std.mem.Allocator, opts: TaskSchedulerOpts) !Value {
        return api.call_global(allocator, "meta_schedule.TaskSchedulerGradientBased", &.{
            opts.logger,
            Value.float(opts.alpha),
            Value.int(opts.window_size),
            Value.int(opts.seed),
        });
    }

    pub const RunTuneOpts = struct {
        scheduler: Value,
        contexts: Value,
        weights: Value,
        max_trials: i64,
        max_trials_global: i64,
        trials_per_iter: i64,
        builder: Value,
        runner: Value,
        callbacks: Value,
        database: Value,
        cost_model: Value,
    };

    /// Run the tuning loop.
    pub fn run_tune(allocator: std.mem.Allocator, opts: RunTuneOpts) !void {
        _ = try api.call_global(allocator, "meta_schedule.TaskSchedulerTune", &.{
            opts.scheduler,
            opts.contexts,
            opts.weights,
            Value.int(opts.max_trials),
            Value.int(opts.max_trials_global),
            Value.int(opts.trials_per_iter),
            opts.builder,
            opts.runner,
            opts.callbacks,
            opts.database,
            opts.cost_model,
        });
    }

    /// Create a BuilderResult (success or error).
    pub fn builder_result(allocator: std.mem.Allocator, artifact_path: ?[:0]const u8, error_msg: ?[:0]const u8) !Value {
        return api.call_global(allocator, "meta_schedule.BuilderResult", &.{
            if (artifact_path) |p| Value.str(p) else Value.none(),
            if (error_msg) |m| Value.str(m) else Value.none(),
        });
    }

    /// Create a RunnerResult (success or error).
    ///
    /// For success: pass `run_secs` as an Array of floats, `error_msg` as null.
    /// For error: pass `run_secs` as null, `error_msg` as the message.
    pub fn runner_result(allocator: std.mem.Allocator, run_secs: ?Value, error_msg: ?[:0]const u8) !Value {
        return api.call_global(allocator, "meta_schedule.RunnerResult", &.{
            run_secs orelse Value.none(),
            if (error_msg) |m| Value.str(m) else Value.none(),
        });
    }

    /// Create a RunnerFuture wrapping a RunnerResult with trivial done/result callbacks.
    pub fn runner_future(allocator: std.mem.Allocator, result: Value) !Value {
        const done_cb = struct {
            fn f(_: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, ret: [*c]c.TVMFFIAny) callconv(.c) c_int {
                ret.* = Value.boolean(true).raw;
                return 0;
            }
        }.f;
        const f_done = try api.create_packed_func(null, done_cb, null);
        defer f_done.decref();

        const ResultHolder = struct {
            raw: c.TVMFFIAny,
            fn callback(self_ptr: ?*anyopaque, _: [*c]const c.TVMFFIAny, _: i32, ret: [*c]c.TVMFFIAny) callconv(.c) c_int {
                const self: *@This() = @ptrCast(@alignCast(self_ptr orelse return -1));
                ret.* = self.raw;
                return 0;
            }
        };
        const holder = try std.heap.c_allocator.create(ResultHolder);
        holder.raw = result.raw;

        const f_result = try api.create_packed_func(@ptrCast(holder), ResultHolder.callback, struct {
            fn dtor(self_ptr: ?*anyopaque) callconv(.c) void {
                const self: *ResultHolder = @ptrCast(@alignCast(self_ptr orelse return));
                std.heap.c_allocator.destroy(self);
            }
        }.dtor);
        defer f_result.decref();

        return api.call_global(allocator, "meta_schedule.RunnerFuture", &.{ f_done, f_result });
    }

    /// Create a MeasureCallbackAddToDatabase callback.
    pub fn add_to_database(allocator: std.mem.Allocator) !Value {
        return api.call_global(allocator, "meta_schedule.MeasureCallbackAddToDatabase", &.{});
    }
};

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
