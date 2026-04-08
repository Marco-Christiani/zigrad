//! JIT: typed compiled function wrapper.
//!
//! Combines trace + compile into a callable that preserves the original
//!  function's structured input/output types. This is the highest-level API
//!  for compiling and executing Zigrad programs.
//!
//! ## Layer 1 usage
//!
//! ```zig
//! var step = try zg.jit(train_step, allocator, backend, device, .{
//!     params_spec, batch_spec,
//! }, .{ .donate = &.{0} });
//! defer step.deinit();
//!
//! // Donated inputs (params) are updated in-place after each call.
//! // Only non-donated outputs are returned.
//! var inputs: @TypeOf(step).InputType = .{ params, batch };
//! const result = try step.call(&inputs);
//! const loss = try result.loss_val.item(f32);
//! result.loss_val.deinit();
//! // inputs[0] now holds the updated params -- no manual reassignment.
//! ```
//!
//! For manual control over trace and compile, use Layer 2:
//!  `zg.trace()` + `zg.frontend.compile_program()`.
const std = @import("std");

const pr = @import("../pr/pr.zig");
const backend_mod = @import("../backend/root.zig");
const Backend = backend_mod.Backend;
const Tensor = @import("../tensor.zig");
const tree = @import("../utils/tree.zig");
const RuntimeOf = tree.RuntimeOf;
const TensorTree = tree.Tree(Tensor);
const frontend = @import("frontend.zig");
const train = @import("train.zig");

const log = std.log.scoped(.@"zg/jit");

pub const JitOpts = struct {
    /// Argument positions to donate (in-place update). Donated arguments'
    ///  buffers are swapped back into the input after execution -- the
    ///  corresponding return fields are stripped from the call result.
    donate: []const usize = &.{},
    /// Options forwarded to the compilation pipeline.
    compile: frontend.CompileOpts = .{},
};

/// Create a typed compiled function from a comptime trace function.
///
/// `func` is the comptime function to trace (same as passed to `zg.trace`).
/// `specs` is a tuple of abstract Tensor specs defining input shapes/dtypes.
/// Returns a `Compiled` instance whose `call()` method accepts a pointer to
///  the structured input type. Donated inputs are updated in-place; the return
///  type excludes donated fields.
///
/// Ownership: the returned `Compiled` owns the compiled executable.
///  Call `deinit()` when done.
pub fn jit(
    comptime func: anytype,
    allocator: std.mem.Allocator,
    backend: *Backend,
    device: Backend.Device,
    specs: anytype,
    comptime opts: JitOpts,
) !Compiled(func, @TypeOf(specs), opts) {
    return Compiled(func, @TypeOf(specs), opts).init(allocator, backend, device, specs);
}

/// Typed compiled function. Parameterized by the traced function, spec tuple
///  type, and JIT options (including donation).
///
/// When donation is active, `call()` accepts a `*InputType` and swaps donated
///  output buffers back into the caller's input struct. The return type is the
///  original return type with donated fields removed -- only non-donated outputs
///  are returned.
pub fn Compiled(
    comptime func: anytype,
    comptime SpecsTuple: type,
    comptime opts: JitOpts,
) type {
    const FullReturnType = DeriveReturnType(func);
    const spec_fields = @typeInfo(SpecsTuple).@"struct".fields;
    const input_count = TensorTree.leaf_count(SpecsTuple);
    const output_count = TensorTree.leaf_count(FullReturnType);
    const non_donatable = comptime train.donate_argnums(SpecsTuple, opts.donate);

    // --- Donation analysis (comptime) ---

    const DonateMap = struct { input_leaf_offset: usize, output_leaf_offset: usize, leaf_count: usize, ret_field_idx: usize };

    const donate_maps: [opts.donate.len]DonateMap = comptime blk: {
        if (opts.donate.len == 0) break :blk .{};

        const ret_fields = @typeInfo(FullReturnType).@"struct".fields;
        var maps: [opts.donate.len]DonateMap = undefined;
        var claimed: [ret_fields.len]bool = @splat(false);

        for (opts.donate, 0..) |d, di| {
            const donated_type = spec_fields[d].type;

            // Input leaf offset for this arg.
            var input_off: usize = 0;
            for (spec_fields[0..d]) |f| input_off += TensorTree.leaf_count(f.type);

            // Find matching unclaimed return field by type.
            var found: ?usize = null;
            for (ret_fields, 0..) |rf, ri| {
                if (rf.type == donated_type and !claimed[ri]) {
                    if (found != null) @compileError(
                        "ambiguous donation: multiple return fields match type " ++ @typeName(donated_type),
                    );
                    found = ri;
                }
            }
            const fi = found orelse @compileError(
                "donated arg type " ++ @typeName(donated_type) ++
                    " not found in return type -- donated args must appear in the return for buffer replacement",
            );
            claimed[fi] = true;

            var output_off: usize = 0;
            for (ret_fields[0..fi]) |rf| output_off += TensorTree.leaf_count(rf.type);

            maps[di] = .{
                .input_leaf_offset = input_off,
                .output_leaf_offset = output_off,
                .leaf_count = TensorTree.leaf_count(donated_type),
                .ret_field_idx = fi,
            };
        }
        break :blk maps;
    };

    // Per-output-leaf mask: true = kept (goes to output_tensors), false = donated (goes back to input).
    const kept_mask: [output_count]bool = comptime blk: {
        var mask: [output_count]bool = @splat(true);
        for (donate_maps) |dm| {
            for (dm.output_leaf_offset..dm.output_leaf_offset + dm.leaf_count) |i| {
                mask[i] = false;
            }
        }
        break :blk mask;
    };

    const kept_output_count: usize = comptime blk: {
        var n: usize = 0;
        for (kept_mask) |k| {
            if (k) n += 1;
        }
        break :blk n;
    };

    // Return type with donated fields stripped.
    const CallReturn = comptime blk: {
        if (opts.donate.len == 0) break :blk RuntimeOf(FullReturnType);

        const ret_info = @typeInfo(FullReturnType).@"struct";
        const kept_field_count = ret_info.fields.len - opts.donate.len;

        if (kept_field_count == 0) break :blk void;

        // Collect indices of donated return fields for fast lookup.
        var donated_indices: [opts.donate.len]usize = undefined;
        for (donate_maps, 0..) |dm, i| donated_indices[i] = dm.ret_field_idx;

        var fields: [kept_field_count]std.builtin.Type.StructField = undefined;
        var idx: usize = 0;
        for (ret_info.fields, 0..) |field, fi| {
            var is_donated = false;
            for (donated_indices) |di| {
                if (fi == di) {
                    is_donated = true;
                    break;
                }
            }
            if (!is_donated) {
                fields[idx] = .{
                    .name = field.name,
                    .type = field.type,
                    .default_value_ptr = null,
                    .is_comptime = false,
                    .alignment = @alignOf(field.type),
                };
                idx += 1;
            }
        }

        break :blk @Type(.{ .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        } });
    };

    return struct {
        const Self = @This();

        /// The structured input type accepted by `call()`.
        pub const InputType = RuntimeOf(SpecsTuple);

        /// The return type of `call()` -- the original return type with
        ///  donated fields removed.
        pub const ReturnType = CallReturn;

        exe: Backend.Executable,
        backend: *Backend,
        /// Pre-built output tensors for kept (non-donated) outputs.
        /// Buffer handles are overwritten after each execute.
        output_tensors: [kept_output_count]Tensor,

        pub fn init(
            allocator: std.mem.Allocator,
            backend_: *Backend,
            device: Backend.Device,
            specs: SpecsTuple,
        ) !Self {
            var program = try frontend.trace(func, allocator, specs, "jit");
            defer program.deinit();

            const entry = program.get_function("jit") orelse return error.NoEntry;
            std.debug.assert(entry.returns.len == output_count);

            // Pre-build output tensors for kept (non-donated) outputs only.
            var output_tensors: [kept_output_count]Tensor = undefined;
            var kept_idx: usize = 0;
            for (entry.returns, 0..) |ret_var, i| {
                if (kept_mask[i]) {
                    const t = ret_var.as_tensor();
                    output_tensors[kept_idx] = .{
                        .dtype = t.dtype,
                        .shape = .from_slice(t.shape.dims),
                        .backing = .{ .device = .{ .buffer = undefined, .backend = backend_ } },
                    };
                    kept_idx += 1;
                }
            }

            const exe = try frontend.compile_program(
                backend_,
                allocator,
                &program,
                device,
                "jit",
                opts.compile,
            );

            return .{
                .exe = exe,
                .backend = backend_,
                .output_tensors = output_tensors,
            };
        }

        /// Execute the compiled function with structured inputs.
        ///
        /// Takes a pointer to the input struct. After execution:
        ///
        /// 1. Donated inputs are updated in-place -- their buffer handles are
        ///    replaced with the corresponding output buffers. Old buffers are
        ///    freed if the backend allocated new ones (handles differ).
        /// 2. Non-donated outputs are returned in a struct with the donated
        ///    fields removed.
        ///
        /// The caller owns all returned Tensors (call `.deinit()` on each).
        pub fn call(self: *Self, args: *InputType) !CallReturn {
            // Walk input struct at comptime, extract buffers directly.
            var input_bufs: [input_count]Backend.Buffer = undefined;
            var in_idx: usize = 0;
            try collect_buffers(InputType, args.*, &input_bufs, &in_idx);

            // Execute.
            var output_bufs: [output_count]Backend.Buffer = undefined;
            const event = try self.backend.execute_into(
                self.exe,
                &input_bufs,
                &output_bufs,
                non_donatable,
                .{},
            );
            if (event) |ev| self.backend.deinit_event(ev);

            // Route kept output buffers into pre-built output tensors.
            var kept_idx: usize = 0;
            inline for (0..output_count) |i| {
                if (kept_mask[i]) {
                    self.output_tensors[kept_idx].backing.device.buffer = output_bufs[i];
                    kept_idx += 1;
                }
            }

            // Swap donated output buffers back into the caller's input tensors.
            // Frees old buffer handles when the backend allocated new ones.
            inline for (0..opts.donate.len) |di| {
                const dm = donate_maps[di];
                var buf_idx: usize = dm.output_leaf_offset;
                swap_donated_buffers(
                    spec_fields[opts.donate[di]].type,
                    &@field(args, spec_fields[opts.donate[di]].name),
                    &output_bufs,
                    &buf_idx,
                    self.backend,
                );
            }

            if (CallReturn == void) return;
            return TensorTree.unflatten(CallReturn, &self.output_tensors);
        }

        /// Comptime-recursive walk: extract device buffers from a nested
        ///  struct of Tensors into a flat buffer array.
        /// TODO: This is essentially just tree.flatten_values, but we extract a field,
        ///  honestly, not neccessary... but flatten_values is private and flatten
        ///  allocates. make it pub?
        fn collect_buffers(comptime T: type, value: T, out: []Backend.Buffer, idx: *usize) !void {
            if (T == Tensor) {
                out[idx.*] = try value.buffer();
                idx.* += 1;
                return;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    inline for (info.fields) |field| {
                        try collect_buffers(field.type, @field(value, field.name), out, idx);
                    }
                },
                .array => |info| {
                    inline for (0..info.len) |i| {
                        try collect_buffers(info.child, value[i], out, idx);
                    }
                },
                else => @compileError("unsupported type in jit input: " ++ @typeName(T)),
            }
        }

        /// Comptime-recursive walk: swap output buffers back into donated
        ///  input tensors. Frees old handles when the backend allocated new ones
        ///  (handle differs), matching `TrainState.step` semantics.
        fn swap_donated_buffers(
            comptime T: type,
            target: *T,
            bufs: []const Backend.Buffer,
            idx: *usize,
            backend_: *Backend,
        ) void {
            if (T == Tensor) {
                const new_buf = bufs[idx.*];
                const old_buf = target.backing.device.buffer;
                if (new_buf.handle != old_buf.handle) {
                    backend_.deinit_buffer(old_buf);
                }
                target.backing.device.buffer = new_buf;
                idx.* += 1;
                return;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    inline for (info.fields) |field| {
                        swap_donated_buffers(field.type, &@field(target, field.name), bufs, idx, backend_);
                    }
                },
                .array => |info| {
                    inline for (0..info.len) |i| {
                        swap_donated_buffers(info.child, &target[i], bufs, idx, backend_);
                    }
                },
                else => @compileError("unsupported type in donated input: " ++ @typeName(T)),
            }
        }

        /// Release all device buffers held by an input struct.
        ///
        /// Call after the last `call()` to free the final input buffers
        ///  (donated args from the last step + non-donated args).
        pub fn deinit_inputs(self: *Self, args: *InputType) void {
            deinit_tensors(InputType, args);
            _ = self; // backend not needed -- Tensor.deinit handles it
        }

        fn deinit_tensors(comptime T: type, target: *T) void {
            if (T == Tensor) {
                target.deinit();
                return;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    inline for (info.fields) |field| {
                        deinit_tensors(field.type, &@field(target, field.name));
                    }
                },
                .array => |info| {
                    inline for (0..info.len) |i| {
                        deinit_tensors(info.child, &target[i]);
                    }
                },
                else => @compileError("unsupported type in jit input: " ++ @typeName(T)),
            }
        }

        pub fn deinit(self: *Self) void {
            self.backend.deinit_executable(self.exe);
        }
    };
}

/// Derive the return type of a traced function from its comptime signature.
fn DeriveReturnType(comptime func: anytype) type {
    const FnInfo = @typeInfo(@TypeOf(func)).@"fn";
    const RawReturn = FnInfo.return_type.?;

    // Unwrap error union if present.
    return switch (@typeInfo(RawReturn)) {
        .error_union => |eu| eu.payload,
        else => RawReturn,
    };
}
