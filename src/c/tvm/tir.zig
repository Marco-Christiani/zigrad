//! TVM IR, TIR, and Target wrappers.
//!
//! Covers the `tvm/ir/`, `tvm/tir/`, `tvm/target/`, and `tvm/te/` operations
//!  required by TIR lowering.
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const dlpack = @import("../dlpack.zig");
const Value = api.Value;
const ObjectHandle = api.ObjectHandle;
const TvmError = api.TvmError;
const TargetKind = @import("../../tvm/config.zig").TargetKind;

const helpers = api.helpers;
const log = std.log.scoped(.@"zg/tvm_tir");

// IRModule

pub const IRModule = struct {
    handle: ObjectHandle,
    /// TVM runtime type index, preserved from the FFI call that created this object.
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    pub const deinit = helpers.deinit(IRModule);
    pub const as_value = helpers.as_value(IRModule);

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
        log.debug("applied {s} (ptr {s}, type_index {d}{s}{d})", .{
            pass.name(),
            if (old_ptr != new_obj) "changed" else "same",
            @as(c_int, if (old_ptr == new_obj) self.type_index else 0),
            "->",
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

// Target

pub const Target = struct {
    handle: ObjectHandle,
    type_index: c_int = c.kTVMFFIStaticObjectBegin,

    pub const deinit = helpers.deinit(Target);
    pub const as_value = helpers.as_value(Target);

    /// Create a Target from a TargetKind.
    ///
    /// For CPU targets, includes `-num-cores` (required by MetaSchedule).
    pub fn create(
        allocator: std.mem.Allocator,
        kind: TargetKind,
        device_ordinal: i32,
    ) !Target {
        var resolution = try describe(
            allocator,
            kind,
            device_ordinal,
        );
        defer resolution.deinit();
        return try create_from_description(allocator, resolution.description);
    }

    /// Detect a target and retain its cache identity with the TVM object.
    pub fn resolve(
        allocator: std.mem.Allocator,
        kind: TargetKind,
        device_ordinal: i32,
    ) !ResolvedTarget {
        var resolution = try describe(
            allocator,
            kind,
            device_ordinal,
        );
        errdefer resolution.deinit();
        log.info(
            "resolved {s} target for device {d}: {s}",
            .{ @tagName(kind), device_ordinal, resolution.description },
        );
        return .{
            .allocator = allocator,
            .description = resolution.description,
            .gpu_arch = resolution.gpu_arch,
            .target = try create_from_description(
                allocator,
                resolution.description,
            ),
        };
    }

    /// Detect the target properties used for compilation and cache identity.
    pub fn describe(
        allocator: std.mem.Allocator,
        kind: TargetKind,
        device_ordinal: i32,
    ) !TargetDescription {
        return switch (kind) {
            .cpu => blk: {
                const cpu_count = std.Thread.getCpuCount() catch 1;
                break :blk .{
                    .allocator = allocator,
                    .description = try std.fmt.allocPrintSentinel(
                        allocator,
                        "llvm -num-cores={d}",
                        .{cpu_count},
                        0,
                    ),
                    .gpu_arch = null,
                };
            },
            .cuda => blk: {
                const properties = try detect_cuda_properties(
                    allocator,
                    device_ordinal,
                );
                defer allocator.free(properties.compute_version);
                const gpu_arch = try cuda_architecture(
                    allocator,
                    properties.compute_version,
                );
                errdefer allocator.free(gpu_arch);
                break :blk .{
                    .allocator = allocator,
                    .description = try cuda_target_description(
                        allocator,
                        properties,
                        gpu_arch,
                    ),
                    .gpu_arch = gpu_arch,
                };
            },
        };
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

/// TVM target paired with the description that produced it.
pub const ResolvedTarget = struct {
    allocator: std.mem.Allocator,
    description: [:0]u8,

    /// NVRTC architecture spelling for CUDA targets.
    gpu_arch: ?[]u8,

    target: Target,

    pub fn deinit(self: *ResolvedTarget) void {
        self.target.deinit();
        self.allocator.free(self.description);
        if (self.gpu_arch) |gpu_arch| self.allocator.free(gpu_arch);
        self.* = undefined;
    }
};

const CudaProperties = struct {
    compute_version: []const u8,
    max_threads_per_block: i64,
    thread_warp_size: i64,
    max_shared_memory_per_block: i64,
    registers_per_block: i64,
    l2_cache_size_bytes: i64,
};

fn create_from_description(
    allocator: std.mem.Allocator,
    description: [:0]const u8,
) !Target {
    const result = try api.call_global(
        allocator,
        "target.Target",
        &.{Value.str(description)},
    );
    const obj = result.as_object() orelse return error.TvmCallFailed;
    return .{ .handle = .{ .ptr = obj }, .type_index = result.raw.type_index };
}

pub const TargetDescription = struct {
    allocator: std.mem.Allocator,
    description: [:0]u8,
    gpu_arch: ?[]u8,

    pub fn deinit(self: *TargetDescription) void {
        self.allocator.free(self.description);
        if (self.gpu_arch) |gpu_arch| self.allocator.free(gpu_arch);
        self.* = undefined;
    }
};

fn detect_cuda_properties(
    allocator: std.mem.Allocator,
    device_ordinal: i32,
) !CudaProperties {
    if (device_ordinal < 0) return error.TvmCallFailed;
    if (try device_int_attribute(device_ordinal, .exist) == 0) {
        log.err("TVM cannot access CUDA device {d}", .{device_ordinal});
        return error.TvmCallFailed;
    }

    const compute_version = try device_string_attribute(
        allocator,
        device_ordinal,
        .compute_version,
    );
    errdefer allocator.free(compute_version);

    return .{
        .compute_version = compute_version,
        .max_threads_per_block = try device_int_attribute(
            device_ordinal,
            .max_threads_per_block,
        ),
        .thread_warp_size = try device_int_attribute(device_ordinal, .warp_size),
        .max_shared_memory_per_block = try device_int_attribute(
            device_ordinal,
            .max_shared_memory_per_block,
        ),
        .registers_per_block = try device_int_attribute(
            device_ordinal,
            .max_registers_per_block,
        ),
        .l2_cache_size_bytes = try device_int_attribute(
            device_ordinal,
            .l2_cache_size_bytes,
        ),
    };
}

const DeviceAttribute = enum(i64) {
    exist = 0,
    max_threads_per_block = 1,
    warp_size = 2,
    max_shared_memory_per_block = 3,
    compute_version = 4,
    max_registers_per_block = 9,
    l2_cache_size_bytes = 13,
};

fn device_int_attribute(
    device_ordinal: i32,
    attribute: DeviceAttribute,
) !i64 {
    const result = try api.call_global(
        std.heap.c_allocator,
        "runtime.GetDeviceAttr",
        &.{
            Value.int(@intFromEnum(dlpack.DeviceType.cuda)),
            Value.int(device_ordinal),
            Value.int(@intFromEnum(attribute)),
        },
    );
    return result.to_int() orelse error.UnexpectedTvmType;
}

fn device_string_attribute(
    allocator: std.mem.Allocator,
    device_ordinal: i32,
    attribute: DeviceAttribute,
) ![]u8 {
    var result = try api.call_global(
        allocator,
        "runtime.GetDeviceAttr",
        &.{
            Value.int(@intFromEnum(dlpack.DeviceType.cuda)),
            Value.int(device_ordinal),
            Value.int(@intFromEnum(attribute)),
        },
    );
    return try result.as_string(allocator);
}

fn cuda_target_description(
    allocator: std.mem.Allocator,
    properties: CudaProperties,
    gpu_arch: []const u8,
) ![:0]u8 {
    if (gpu_arch.len == 0) return error.TvmCallFailed;

    return try std.fmt.allocPrintSentinel(
        allocator,
        "cuda -arch={s} -max_shared_memory_per_block={d} " ++
            "-max_threads_per_block={d} -thread_warp_size={d} " ++
            "-registers_per_block={d} -l2_cache_size_bytes={d}",
        .{
            gpu_arch,
            properties.max_shared_memory_per_block,
            properties.max_threads_per_block,
            properties.thread_warp_size,
            properties.registers_per_block,
            properties.l2_cache_size_bytes,
        },
        0,
    );
}

fn cuda_architecture(
    allocator: std.mem.Allocator,
    compute_version: []const u8,
) ![]u8 {
    var architecture: [16]u8 = undefined;
    var architecture_len: usize = 0;
    for (compute_version) |byte| switch (byte) {
        '0'...'9' => {
            if (architecture_len == architecture.len) return error.TvmCallFailed;
            architecture[architecture_len] = byte;
            architecture_len += 1;
        },
        '.' => {},
        else => return error.TvmCallFailed,
    };
    if (architecture_len == 0) return error.TvmCallFailed;
    return try std.fmt.allocPrint(
        allocator,
        "sm_{s}",
        .{architecture[0..architecture_len]},
    );
}

test cuda_target_description {
    const description = try cuda_target_description(
        std.testing.allocator,
        .{
            .compute_version = "8.9",
            .max_threads_per_block = 1024,
            .thread_warp_size = 32,
            .max_shared_memory_per_block = 49152,
            .registers_per_block = 65536,
            .l2_cache_size_bytes = 75497472,
        },
        "sm_89",
    );
    defer std.testing.allocator.free(description);

    try std.testing.expectEqualStrings(
        "cuda -arch=sm_89 -max_shared_memory_per_block=49152 " ++
            "-max_threads_per_block=1024 -thread_warp_size=32 " ++
            "-registers_per_block=65536 -l2_cache_size_bytes=75497472",
        description,
    );
}

test cuda_architecture {
    const architecture = try cuda_architecture(std.testing.allocator, "8.9");
    defer std.testing.allocator.free(architecture);
    try std.testing.expectEqualStrings("sm_89", architecture);
}

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
    unroll_loop,
    renormalize_split_pattern,

    // Passes with arguments
    filter: struct { predicate: Value },
    bind_target: struct { target: Target },
    thread_sync: struct { scope: [:0]const u8 },
    compact_buffer_alloc: struct { is_strict: bool },
    narrow_data_type: struct { target_bits: i64 },
    vectorize_loop: struct { enable: bool },
    common_subexpr_elim: struct { enable_cse: bool, enable_equiv: bool },

    /// Returns the TVM global function name for this pass.
    ///
    /// Most names follow `"tir.transform." ++ PascalCase(@tagName)`. Variants
    /// where TVM's name diverges from that convention have explicit overrides.
    pub fn name(self: TirPass) []const u8 {
        return switch (self) {
            // Overrides where TVM name diverges from PascalCase(@tagName)
            .filter => "tir.transform.Filter",
            .plan_and_update_buffer_allocation => "tir.transform.PlanAndUpdateBufferAllocationLocation",
            .compact_buffer_alloc => "tir.transform.CompactBufferAllocation",
            .common_subexpr_elim => "tir.transform.CommonSubexprElimTIR",
            .make_packed_api => "tir.transform.MakePackedAPI",
            .lower_tvm_builtin => "tir.transform.LowerTVMBuiltin",
            inline else => |_, tag| comptime tag_to_pass_name(@tagName(tag)),
        };
    }

    /// Create the TVM pass object by calling the global function with args.
    pub fn create(self: TirPass, allocator: std.mem.Allocator) TvmError!Value {
        return switch (self) {
            .filter => |args| try api.call_global(allocator, self.name(), &.{args.predicate}),
            .bind_target => |args| try api.call_global(allocator, self.name(), &.{args.target.as_value()}),
            .thread_sync => |args| try api.call_global(allocator, self.name(), &.{Value.str(args.scope)}),
            .compact_buffer_alloc => |args| try api.call_global(allocator, self.name(), &.{Value.boolean(args.is_strict)}),
            .narrow_data_type => |args| try api.call_global(allocator, self.name(), &.{Value.int(args.target_bits)}),
            .vectorize_loop => |args| try api.call_global(allocator, self.name(), &.{Value.boolean(args.enable)}),
            .common_subexpr_elim => |args| try api.call_global(allocator, self.name(), &.{
                Value.boolean(args.enable_cse),
                Value.boolean(args.enable_equiv),
            }),
            else => try api.call_global(allocator, self.name(), &.{}),
        };
    }

    /// Comptime: "tir.transform." ++ snake_to_pascal(tag_name).
    fn tag_to_pass_name(comptime tag_name: [:0]const u8) [:0]const u8 {
        @setEvalBranchQuota(5000);
        const result = comptime blk: {
            const prefix = "tir.transform.";
            var underscores: usize = 0;
            for (tag_name) |ch| {
                if (ch == '_') underscores += 1;
            }
            const total_len = prefix.len + tag_name.len - underscores;
            var buf: [total_len:0]u8 = undefined;
            for (prefix, 0..) |ch, i| buf[i] = ch;
            var ri: usize = prefix.len;
            var cap_next = true;
            for (tag_name) |ch| {
                if (ch == '_') {
                    cap_next = true;
                } else {
                    buf[ri] = if (cap_next) (ch - 32) else ch;
                    ri += 1;
                    cap_next = false;
                }
            }
            break :blk buf;
        };
        return &result;
    }
};

// Function attribute access

/// Get the attribute Map from an IR function (PrimFunc, etc.).
///
/// Calls `ir.BaseFunc_Attrs` then `ir.DictAttrsGetDict` and wraps the
/// result as a Map. Returns null if the function has no attributes.
pub fn get_func_attrs(allocator: std.mem.Allocator, func: Value) TvmError!?api.Map {
    const dict_attrs = try api.call_global(allocator, "ir.BaseFunc_Attrs", &.{func});
    defer dict_attrs.decref();
    if (dict_attrs.raw.type_index == c.kTVMFFINone) return null;

    const map_val = try api.call_global(allocator, "ir.DictAttrsGetDict", &.{dict_attrs});
    const obj = map_val.as_object() orelse return TvmError.TvmCallFailed;
    return .{ .handle = .{ .ptr = obj }, .type_index = map_val.raw.type_index };
}

/// Construct a TensorIntrin from description and implementation PrimFuncs.
pub fn tensor_intrin(allocator: std.mem.Allocator, desc: Value, impl: Value) TvmError!Value {
    return try api.call_global(allocator, "tir.TensorIntrin", &.{ desc, impl });
}

/// Register a tensor intrinsic by name.
pub fn register_tensor_intrin(allocator: std.mem.Allocator, name: [:0]const u8, intrin_val: Value, override: bool) TvmError!void {
    _ = try api.call_global(allocator, "tir.TensorIntrinRegister", &.{
        Value.str(name), intrin_val, Value.boolean(override),
    });
}

/// Build a matmul IRModule from shapes via TE (topi.matmul).
///
/// Creates A[M,K] @ B[K,N] = C[M,N] via te.Placeholder + topi.matmul,
/// wraps in CreatePrimFunc + IRModule with global_symbol="main".
///
/// TODO(tvm): Move this fixed matmul recipe behind the kernel-provider
///  compilation operation.
pub fn build_matmul_tir(allocator: std.mem.Allocator, m: i64, n: i64, k: i64) TvmError!IRModule {
    const shape_a = try api.call_global(allocator, "ffi.Array", &.{ Value.int(m), Value.int(k) });
    defer shape_a.decref();
    const shape_b = try api.call_global(allocator, "ffi.Array", &.{ Value.int(k), Value.int(n) });
    defer shape_b.decref();

    const tensor_a = try api.call_global(allocator, "te.Placeholder", &.{
        shape_a, Value.str("float32"), Value.str("A"),
    });
    const tensor_b = try api.call_global(allocator, "te.Placeholder", &.{
        shape_b, Value.str("float32"), Value.str("B"),
    });
    log.debug("created placeholders A[{d},{d}] B[{d},{d}]", .{ m, k, k, n });

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
    const prim_func_attr = try api.call_global(allocator, "ir.BaseFuncWithAttr", &.{
        prim_func, Value.str("global_symbol"), Value.str("main"),
    });
    prim_func.decref();

    // Wrap in IRModule
    const global_var = try api.call_global(allocator, "ir.GlobalVar", &.{Value.str("main")});
    const func_map = try api.call_global(allocator, "ffi.Map", &.{ global_var, prim_func_attr });
    const empty_map = try api.call_global(allocator, "ffi.Map", &.{});
    const ir_mod = try api.call_global(allocator, "ir.IRModule", &.{ func_map, Value.none(), empty_map });

    const obj = ir_mod.as_object() orelse return TvmError.TvmCallFailed;
    log.info("created matmul IRModule ({d}x{d}x{d})", .{ m, n, k });
    return .{ .handle = .{ .ptr = obj }, .type_index = ir_mod.raw.type_index };
}
