//! TVM IR, TIR, and Target wrappers.
//!
//! Covers `tvm/ir/`, `tvm/tir/`, `tvm/target/`, and `tvm/te/` — types that
//! are tightly coupled in TIR lowering but individually too small to warrant
//! separate files at our current usage level.
const std = @import("std");
const api = @import("api.zig");
const c = @import("c.zig");
const Value = api.Value;
const ObjectHandle = api.ObjectHandle;
const TvmError = api.TvmError;

const helpers = api.helpers;
const log = std.log.scoped(.@"zg/tvm_tir");

/// Target kind for TVM compilation.
pub const TargetKind = enum { cpu, cuda };

// ============================================================================
// IRModule
// ============================================================================

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

    pub const deinit = helpers.deinit(Target);
    pub const as_value = helpers.as_value(Target);

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
    ///
    /// Most names follow `"tir.transform." ++ PascalCase(@tagName)`. Variants
    /// where TVM's name diverges from that convention have explicit overrides.
    pub fn name(self: TirPass) []const u8 {
        return switch (self) {
            // Overrides where TVM name diverges from PascalCase(@tagName)
            .plan_and_update_buffer_allocation => "tir.transform.PlanAndUpdateBufferAllocationLocation",
            .compact_buffer_alloc => "tir.transform.CompactBufferAllocation",
            .common_subexpr_elim => "tir.transform.CommonSubexprElimTIR",
            .make_packed_api => "tir.transform.MakePackedAPI",
            .lower_tvm_builtin => "tir.transform.LowerTVMBuiltin",
            inline else => |_, tag| comptime tvmPassName(@tagName(tag)),
        };
    }

    /// Create the TVM pass object by calling the global function with args.
    fn create(self: TirPass, allocator: std.mem.Allocator) TvmError!Value {
        return switch (self) {
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
            else => try api.call_global(allocator, self.name(), &.{}),
        };
    }

    /// Comptime: "tir.transform." ++ snakeToPascal(tag_name).
    fn tvmPassName(comptime tag_name: [:0]const u8) [:0]const u8 {
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
