//! Binding between traced functions and loaded programs.
//!
//! A traced callable exposes its PR program for explicit transformation,
//!  lowering, and compilation. Binding a loaded program restores the original
//!  function's structured input and output types.
//!
//! ## Usage
//!
//! ```zig
//! var traced = try zg.trace_callable(
//!     train_step,
//!     allocator,
//!     .{ params_spec, batch_spec },
//!     .{ .donate = &.{0} },
//! );
//! defer traced.deinit();
//!
//! // Application composition supplies operations for the selected target.
//! var pipeline = zg.compilation.Pipeline.init(allocator);
//! defer pipeline.deinit();
//! try pipeline.add(validate);
//! try pipeline.add(lower_to_target);
//! try pipeline.add(terminal_backend);
//! const loaded_program = try pipeline.run(
//!     zg.Executor.LoadedProgram,
//!     &traced.program,
//!     &context,
//! );
//! var step = try traced.bind(loaded_program);
//! defer step.deinit();
//!
//! // Donated inputs (params) are updated in-place after each call.
//! // Only non-donated outputs are returned.
//! var inputs: @TypeOf(step).InputType = .{ params, batch };
//! const result = try step.call(&inputs);
//! const loss = try result.loss_val.item(f32);
//! result.loss_val.deinit();
//! // inputs[0] now holds the updated parameters.
//! ```
//!
const std = @import("std");

const pr = @import("pr/pr.zig");
const Executor = @import("execution.zig");
const Tensor = @import("tensor.zig");
const utils = @import("utils.zig");
const meta = utils.meta;
const RuntimeOf = utils.RuntimeOf;
const TensorTree = utils.Tree(Tensor);
const trace = @import("trace.zig").trace;

/// Tracing and donation options for `trace_callable`.
pub const Options = struct {
    /// Name assigned to the traced PR entry function.
    entry_name: []const u8 = "main",

    /// Argument positions to donate for in-place updates.
    ///
    /// Donated argument buffers are swapped back into the input after
    ///  execution. Their return fields are stripped from the call result.
    donate: []const usize = &.{},
};

/// Traces a function while retaining its call contract.
///
/// `Traced.deinit` releases the PR program. `Traced.bind` accepts a loaded
///  program after the caller compiles that program.
pub fn trace_callable(
    comptime func: anytype,
    allocator: std.mem.Allocator,
    specs: anytype,
    comptime opts: Options,
) !Traced(func, @TypeOf(specs), opts) {
    return .{
        .program = try trace(func, allocator, specs, opts.entry_name),
        .entry_name = opts.entry_name,
    };
}

/// A PR program paired with the function that produced it.
pub fn Traced(
    comptime func: anytype,
    comptime SpecsTuple: type,
    comptime opts: Options,
) type {
    return struct {
        const Self = @This();

        /// Traced PR program released by `deinit`.
        program: pr.Program,

        /// PR entry function associated with the call contract.
        entry_name: []const u8,

        /// Bind a loaded program compiled from this traced program.
        ///
        /// On success, `Compiled.deinit` releases `loaded_program`. `Traced.deinit`
        ///  releases the PR program separately.
        pub fn bind(
            self: *const Self,
            loaded_program: Executor.LoadedProgram,
        ) error{NoEntry}!Compiled(func, SpecsTuple, opts) {
            const entry = self.program.get_function(self.entry_name) orelse return error.NoEntry;
            return Compiled(func, SpecsTuple, opts).init(loaded_program, entry);
        }

        /// Release the traced PR program.
        pub fn deinit(self: *Self) void {
            self.program.deinit();
            self.* = undefined;
        }
    };
}

/// Compiled function parameterized by its trace contract.
///
/// The contract includes the traced function, spec tuple type, and donation
///  options.
///
/// When donation is active, `call()` accepts a `*InputType` and swaps donated
///  output buffers back into the caller's input struct. The return type is the
///  original return type with donated fields removed. Only retained outputs are
///  returned.
pub fn Compiled(
    comptime func: anytype,
    comptime SpecsTuple: type,
    comptime opts: Options,
) type {
    const FullReturnType = DeriveReturnType(func);
    const spec_fields = @typeInfo(SpecsTuple).@"struct".fields;
    const input_count = TensorTree.leaf_count(SpecsTuple);
    const output_count = TensorTree.leaf_count(FullReturnType);

    const DonateMap = struct { input_leaf_offset: usize, output_leaf_offset: usize, leaf_count: usize, ret_field_idx: usize };

    const donate_maps: [opts.donate.len]DonateMap = comptime blk: {
        if (opts.donate.len == 0) break :blk .{};

        const ret_fields = @typeInfo(FullReturnType).@"struct".fields;
        var maps: [opts.donate.len]DonateMap = undefined;
        var claimed: [ret_fields.len]bool = @splat(false);

        for (opts.donate, 0..) |d, di| {
            const donated_type = spec_fields[d].type;

            var input_off: usize = 0;
            for (spec_fields[0..d]) |f| input_off += TensorTree.leaf_count(f.type);

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
                    " not found in return type. Donated args must appear in the return for buffer replacement",
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

    const donated_input_count: usize = comptime blk: {
        var count: usize = 0;
        for (donate_maps) |donation| count += donation.leaf_count;
        break :blk count;
    };

    const donated_input_indices: [donated_input_count]usize = comptime blk: {
        var indices: [donated_input_count]usize = undefined;
        var index: usize = 0;
        for (donate_maps) |donation| {
            for (0..donation.leaf_count) |leaf_index| {
                indices[index] = donation.input_leaf_offset + leaf_index;
                index += 1;
            }
        }
        break :blk indices;
    };

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

    const CallReturn = comptime blk: {
        if (opts.donate.len == 0) break :blk RuntimeOf(FullReturnType);

        const ret_info = @typeInfo(FullReturnType).@"struct";
        const kept_field_count = ret_info.fields.len - opts.donate.len;

        if (kept_field_count == 0) break :blk void;

        var donated_indices: [opts.donate.len]usize = undefined;
        for (donate_maps, 0..) |dm, i| donated_indices[i] = dm.ret_field_idx;

        var field_names: [kept_field_count][:0]const u8 = undefined;
        var field_types: [kept_field_count]type = undefined;
        var field_attrs: [kept_field_count]std.builtin.Type.StructField.Attributes = undefined;
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
                field_names[idx] = field.name;
                field_types[idx] = field.type;
                field_attrs[idx] = .{ .@"align" = @alignOf(field.type) };
                idx += 1;
            }
        }

        break :blk @Struct(.auto, null, &field_names, &field_types, &field_attrs);
    };

    return struct {
        const Self = @This();

        /// The structured input type accepted by `call()`.
        pub const InputType = RuntimeOf(SpecsTuple);

        /// Return type of `call()` with donated fields removed.
        pub const ReturnType = CallReturn;

        /// Failures reported while validating inputs or invoking the loaded program.
        pub const Error = Executor.Error || error{UnsupportedAval};

        loaded_program: Executor.LoadedProgram,

        /// Output tensors retained by the call result.
        ///
        /// Buffer handles are overwritten after each invocation.
        output_tensors: [kept_output_count]Tensor,

        fn init(
            loaded_program: Executor.LoadedProgram,
            entry: pr.Function,
        ) Self {
            std.debug.assert(entry.returns.len == output_count);
            const executor = loaded_program.executor;

            var output_tensors: [kept_output_count]Tensor = undefined;
            var kept_idx: usize = 0;
            for (entry.returns, 0..) |ret_var, i| {
                if (kept_mask[i]) {
                    const t = ret_var.as_tensor();
                    output_tensors[kept_idx] = .{
                        .dtype = t.dtype,
                        .shape = .from_slice(t.shape.dims),
                        .backing = .{ .device = .{ .buffer = undefined, .executor = executor } },
                    };
                    kept_idx += 1;
                }
            }

            return .{
                .loaded_program = loaded_program,
                .output_tensors = output_tensors,
            };
        }

        /// Execute the compiled function with structured inputs.
        ///
        /// Donated input handles are replaced with their corresponding output
        ///  handles. Distinct old buffers are released. Other outputs are returned
        ///  with donated fields removed.
        ///
        /// The caller must call `deinit` on every returned tensor.
        pub fn call(self: *Self, args: *InputType) Error!CallReturn {
            var input_tensors: [input_count]Tensor = undefined;
            var flat_idx: usize = 0;
            meta.flatten(Tensor, InputType, args.*, &input_tensors, &flat_idx);
            std.debug.assert(flat_idx == input_count);

            var input_buffers: [input_count]Executor.Buffer = undefined;
            for (input_tensors, &input_buffers) |tensor, *buffer| {
                const device = switch (tensor.backing) {
                    .device => |device| device,
                    .traced, .host, .abstract => return error.UnsupportedAval,
                };
                if (device.executor != self.loaded_program.executor) return error.InvalidArgument;
                buffer.* = device.buffer;
            }

            var output_buffers: [output_count]Executor.Buffer = undefined;
            const executor = self.loaded_program.executor;
            const event = try executor.invoke(
                self.loaded_program,
                &input_buffers,
                &output_buffers,
                .{ .donated_input_indices = &donated_input_indices },
            );
            if (event) |completion| {
                defer executor.release_event(completion);
                try executor.wait(completion);
            }

            var kept_idx: usize = 0;
            inline for (0..output_count) |i| {
                if (kept_mask[i]) {
                    self.output_tensors[kept_idx].backing.device.buffer = output_buffers[i];
                    kept_idx += 1;
                }
            }

            inline for (0..opts.donate.len) |di| {
                const dm = donate_maps[di];
                var buf_idx: usize = dm.output_leaf_offset;
                swap_donated_buffers(
                    spec_fields[opts.donate[di]].type,
                    &@field(args, spec_fields[opts.donate[di]].name),
                    &output_buffers,
                    &buf_idx,
                    executor,
                );
            }

            if (CallReturn == void) return;
            return TensorTree.unflatten(CallReturn, &self.output_tensors);
        }

        /// Replace donated tensor buffers with the corresponding invocation outputs.
        ///
        /// Distinct input handles are released before replacement.
        fn swap_donated_buffers(
            comptime T: type,
            target: *T,
            buffers: []const Executor.Buffer,
            idx: *usize,
            executor: *Executor,
        ) void {
            if (T == Tensor) {
                const new_buffer = buffers[idx.*];
                const old_buffer = target.backing.device.buffer;
                if (new_buffer.handle != old_buffer.handle) {
                    executor.release(old_buffer);
                }
                target.backing.device.buffer = new_buffer;
                idx.* += 1;
                return;
            }
            switch (@typeInfo(T)) {
                .@"struct" => |info| {
                    inline for (info.fields) |field| {
                        swap_donated_buffers(field.type, &@field(target, field.name), buffers, idx, executor);
                    }
                },
                .array => |info| {
                    inline for (0..info.len) |i| {
                        swap_donated_buffers(info.child, &target[i], buffers, idx, executor);
                    }
                },
                else => @compileError("unsupported type in donated input: " ++ @typeName(T)),
            }
        }

        /// Release all device buffers held by an input struct.
        ///
        /// Call this after the final `call`. It releases final donated buffers and
        ///  every input buffer that was not donated.
        pub fn deinit_inputs(self: *Self, args: *InputType) void {
            // TODO(meta): Let `visit` accept `Tensor.deinit` directly.
            meta.visit(Tensor, InputType, args, struct {
                fn f(t: *Tensor) void {
                    t.*.deinit();
                }
            }.f);
            _ = self;
        }

        /// Release the loaded program.
        pub fn deinit(self: *Self) void {
            self.loaded_program.deinit();
        }
    };
}

/// Derive the return type of a traced function from its comptime signature.
fn DeriveReturnType(comptime func: anytype) type {
    const FnInfo = @typeInfo(@TypeOf(func)).@"fn";
    const RawReturn = FnInfo.return_type.?;

    return switch (@typeInfo(RawReturn)) {
        .error_union => |eu| eu.payload,
        else => RawReturn,
    };
}

test "Compiled.call waits before releasing an execution event" {
    const FakeExecution = struct {
        interface: Executor = .{
            .device = .{ .platform = .cpu },
            .vtable = &vtable,
        },
        waited: bool = false,
        released_event: bool = false,

        const vtable: Executor.VTable = .{
            .upload = upload,
            .download = download,
            .invoke = invoke,
            .await_event = await_event,
            .release_buffer = release_buffer,
            .release_event = release_event,
            .release_program = release_program,
        };

        fn promote(interface: *Executor) *@This() {
            return @fieldParentPtr("interface", interface);
        }

        fn upload(_: *Executor, _: []const u8, _: @import("dtype.zig").DType, _: []const i64) Executor.Error!Executor.Buffer {
            return error.Unsupported;
        }

        fn download(_: *Executor, _: Executor.Buffer, _: []u8) Executor.Error!?Executor.Event {
            return error.Unsupported;
        }

        fn invoke(
            _: *Executor,
            _: *anyopaque,
            _: []const Executor.Buffer,
            outputs: []Executor.Buffer,
            _: Executor.InvokeOptions,
        ) Executor.Error!?Executor.Event {
            outputs[0] = .{ .handle = @ptrFromInt(2) };
            return .{ .handle = @ptrFromInt(3) };
        }

        fn await_event(interface: *Executor, _: Executor.Event) Executor.Error!void {
            promote(interface).waited = true;
        }

        fn release_buffer(_: *Executor, _: Executor.Buffer) void {}

        fn release_event(interface: *Executor, _: Executor.Event) void {
            const self = promote(interface);
            std.debug.assert(self.waited);
            self.released_event = true;
        }

        fn release_program(_: *Executor, _: *anyopaque) void {}
    };
    const Function = struct {
        fn identity(value: Tensor) !Tensor {
            return value;
        }
    };

    var traced = try trace_callable(
        Function.identity,
        std.testing.allocator,
        .{Tensor.abstract(.f32, &.{1})},
        .{},
    );
    defer traced.deinit();

    var execution: FakeExecution = .{};
    var compiled = try traced.bind(.{
        .executor = &execution.interface,
        .handle = @ptrFromInt(1),
    });
    defer compiled.deinit();

    var inputs = .{Tensor.from_buffer(
        &execution.interface,
        .{ .handle = @ptrFromInt(4) },
        .f32,
        &.{1},
    )};
    defer inputs[0].deinit();
    var result = try compiled.call(&inputs);
    defer result.deinit();

    try std.testing.expect(execution.waited);
    try std.testing.expect(execution.released_event);
}
