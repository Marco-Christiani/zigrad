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

        // Run it: transform.RunPass(pass, module) -> module
        const result = try api.call_global(allocator, "transform.RunPass", &.{ pass_val, self.as_value() });

        const new_obj = result.as_object() orelse return error.TvmCallFailed;
        // Replace handle (decref old if different)
        if (self.handle.ptr != new_obj) {
            self.handle.deinit();
        }
        self.handle.ptr = new_obj;
        self.type_index = result.raw.type_index;
    }

    /// Apply a pass, ignoring failure (for optional/non-fatal passes).
    pub fn apply_pass_optional(self: *IRModule, allocator: std.mem.Allocator, pass: TirPass) void {
        self.apply_pass(allocator, pass) catch {};
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
        const result = try api.call_global(allocator, "target.TargetWithHost", &.{
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
        const result = try api.call_global(allocator, "runtime.ModuleLoadFromFile", &.{
            Value.str(path_z), Value.str(""),
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
    // Incref the module since passes replace it
    ir_mod.handle.incref();

    // BindTarget
    try ir_mod.apply_pass(allocator, .{ .bind_target = .{ .target = target } });

    // Core lowering (order matters)
    ir_mod.apply_pass_optional(allocator, .lower_cross_thread_reduction);
    try ir_mod.apply_pass(allocator, .lower_init_block);
    try ir_mod.apply_pass(allocator, .plan_and_update_buffer_allocation);
    try ir_mod.apply_pass(allocator, .convert_blocks_to_opaque);
    ir_mod.apply_pass_optional(allocator, .lift_thread_binding);
    ir_mod.apply_pass_optional(allocator, .{ .compact_buffer_alloc = .{ .is_strict = false } });
    ir_mod.apply_pass_optional(allocator, .lower_match_buffer);
    try ir_mod.apply_pass(allocator, .lower_opaque_block);
    try ir_mod.apply_pass(allocator, .flatten_buffer);

    // Loop transforms
    ir_mod.apply_pass_optional(allocator, .{ .narrow_data_type = .{ .target_bits = 32 } });
    ir_mod.apply_pass_optional(allocator, .loop_partition);
    ir_mod.apply_pass_optional(allocator, .{ .vectorize_loop = .{ .enable = true } });
    ir_mod.apply_pass_optional(allocator, .inject_virtual_thread);
    ir_mod.apply_pass_optional(allocator, .inject_double_buffer);
    ir_mod.apply_pass_optional(allocator, .storage_rewrite);

    try ir_mod.apply_pass(allocator, .simplify);
    ir_mod.apply_pass_optional(allocator, .remove_no_op);
    ir_mod.apply_pass_optional(allocator, .{ .common_subexpr_elim = .{ .enable_cse = true, .enable_equiv = false } });

    // Entry function annotation
    ir_mod.apply_pass_optional(allocator, .verify_memory);
    try ir_mod.apply_pass(allocator, .annotate_entry_func);

    // CUDA-specific pre-SplitHostDevice passes
    if (kind == .cuda) {
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "shared" } });
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "shared.dyn" } });
        ir_mod.apply_pass_optional(allocator, .{ .thread_sync = .{ .scope = "warp" } });
        ir_mod.apply_pass_optional(allocator, .infer_fragment);
        ir_mod.apply_pass_optional(allocator, .lower_thread_allreduce);
        try ir_mod.apply_pass(allocator, .annotate_device_regions);
    }

    try ir_mod.apply_pass(allocator, .split_host_device);

    if (kind == .cuda) {
        ir_mod.apply_pass_optional(allocator, .merge_shared_memory_allocations);
    }

    try ir_mod.apply_pass(allocator, .make_packed_api);
    ir_mod.apply_pass_optional(allocator, .lower_device_kernel_launch);

    // Finalization — target-specific
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
            // Filter + finalize host and device separately, then build and link.
            return try build_cuda_module(allocator, ir_mod.*, target);
        },
    }
}

/// Build a (CPU) RuntimeModule from a lowered IRModule.
fn build_module(allocator: std.mem.Allocator, ir_mod: IRModule, target: Target) !RuntimeModule {
    const result = try api.call_global(allocator, "target.build", &.{ ir_mod.as_value(), target.as_value() });
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
    const device_built = try api.call_global(allocator, "target.build", &.{ device_mod.as_value(), target.as_value() });
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
    const host_built = try api.call_global(allocator, "target.build", &.{ host_mod.as_value(), host_target.as_value() });

    // Link device into host
    _ = try api.call_global(allocator, "ffi.ModuleImport", &.{ host_built, device_built });

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
