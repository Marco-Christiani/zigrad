const std = @import("std");

const Backend = @import("../Backend.zig");
const Tensor = @import("../tensor.zig");
const TensorTree = @import("../utils.zig").Tree(Tensor);

/// Manages device buffer state across training steps.
///
/// `TrainState` is the execute-side counterpart to `frontend.trace` +
///  `frontend.compile_program`. After tracing and compiling a train step
///  (typically a function that uses `transforms.value_and_grad` + an
///  optimizer), `TrainState` handles the repetitive buffer plumbing:
///
/// - **Donation-aware buffer swap**: donatable inputs (trainable parameters)
///    are automatically replaced with their updated outputs each step. Old
///    buffers are freed.
/// - **Borrowed batch slots**: non-donatable inputs (batch data) are borrowed
///    from the caller, who replaces them each step via `set_batch` or
///    `set_batch_buf`.
/// - **Execute orchestration**: `step()` calls `backend.execute_into`,
///    extracts the loss buffer, and performs the swap.
///
/// ## Typical usage
///
/// ```zig
/// var program = try zg.trace(train_step, allocator, specs, "train_step");
/// defer program.deinit();
/// var exe = try zg.frontend.compile_program(backend, allocator, &program, device, "train_step", .{});
/// defer backend.deinit_executable(exe);
/// var state = try TrainState.init(
///     allocator,
///     exe,
///     backend,
///     initial_tensors,
///     program.output_arity("train_step"),
///     .{ .non_donatable_input_indices = comptime donate_argnums(@TypeOf(specs), &.{0}) },
/// );
/// defer state.deinit(.all);
/// for (0..num_steps) |_| {
///     state.set_batch(batch_tensors);
///     const result = try state.step();
///     const loss_val = try result.loss.item(f32);
///     result.loss.deinit();
/// }
/// ```
pub const TrainState = struct {
    input_bufs: []Backend.Buffer,
    output_bufs: []Backend.Buffer,
    exe: Backend.Executable,
    backend: *Backend,
    /// Number of donatable (owned) inputs. These occupy `input_bufs[0..donatable_count]`.
    /// Non-donatable (borrowed) inputs occupy `input_bufs[donatable_count..]`.
    donatable_count: usize,
    non_donatable_indices: []const i64,
    /// DType of the loss output (output[0]). Used to wrap the raw buffer
    ///  as a Tensor in `StepResult`.
    loss_dtype: @import("../pr/pr.zig").DType,
    allocator: std.mem.Allocator,

    pub const StepResult = struct {
        loss: Tensor,
        event: ?Backend.Event,
    };

    /// Initialize from individual components.
    ///
    /// `initial_tensors` must be in the same flattened order as the spec tree
    ///  passed to `trace`. All tensors must be device-backed.
    ///
    /// Donation is specified via `non_donatable_input_indices` in opts. Use the
    ///  `donate_argnums` helper to compute these from the spec tuple type and
    ///  donated argument positions. Donatable inputs must precede non-donatable
    ///  inputs in the flattened spec order (this is the natural layout when
    ///  params are the first argument and batch data follows).
    ///
    /// Ownership follows the donation split: donatable buffers are owned by
    ///  TrainState (freed on swap/deinit), non-donatable buffers are borrowed
    ///  (caller manages lifetime).
    pub const InitOpts = struct {
        /// Flat indices of non-donatable inputs. Must be sorted ascending.
        /// Use `donate_argnums` to compute from spec types and argument positions.
        non_donatable_input_indices: []const i64 = &.{},
        /// DType of the loss output (output[0]). Defaults to f32.
        /// Override for mixed-precision training (e.g. bf16 loss with
        ///  f32 upcast).
        loss_dtype: @import("../pr/pr.zig").DType = .f32,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        exe: Backend.Executable,
        backend_: *Backend,
        initial_tensors: []const Tensor,
        output_arity: usize,
        opts: InitOpts,
    ) !TrainState {
        const input_bufs = try allocator.alloc(Backend.Buffer, initial_tensors.len);
        errdefer allocator.free(input_bufs);
        for (input_bufs, initial_tensors) |*slot, t| {
            slot.* = try t.buffer();
        }

        const output_bufs = try allocator.alloc(Backend.Buffer, output_arity);
        errdefer allocator.free(output_bufs);
        const donatable_count = initial_tensors.len - opts.non_donatable_input_indices.len;

        return .{
            .input_bufs = input_bufs,
            .output_bufs = output_bufs,
            .exe = exe,
            .backend = backend_,
            .donatable_count = donatable_count,
            .non_donatable_indices = opts.non_donatable_input_indices,
            .loss_dtype = opts.loss_dtype,
            .allocator = allocator,
        };
    }

    /// Execute one training step. Swaps donatable buffers in-place.
    ///
    /// Returns `StepResult` with the loss as a device Tensor and optional
    ///  completion event. Caller owns the loss tensor (call `deinit` or
    ///  use `item()` then `deinit`). Donatable input buffers are replaced
    ///  with their corresponding outputs; old buffers are freed.
    pub fn step(self: *TrainState) !StepResult {
        const event = try self.backend.execute_into(
            self.exe,
            self.input_bufs,
            self.output_bufs,
            self.non_donatable_indices,
            .{},
        );

        const loss_buf = self.output_bufs[0];

        // Swap donatable buffers: output[1+i] replaces input[i] for each
        //  donatable input. Output index is offset by 1 (loss is output[0]).
        for (self.input_bufs[0..self.donatable_count], 1..) |*old, out_idx| {
            const new_buf = self.output_bufs[out_idx];
            if (new_buf.handle != old.handle) {
                self.backend.deinit_buffer(old.*);
                old.* = new_buf;
            }
        }

        return .{
            .loss = Tensor.from_buffer(self.backend, loss_buf, self.loss_dtype, &.{}),
            .event = event,
        };
    }

    /// Replace a non-donatable (borrowed) input by flat index into the
    ///  non-donatable slice (not the full input array).
    ///
    /// The tensor must be device-backed. The old buffer is NOT freed
    ///  (caller owns non-donatable buffers).
    pub fn set_batch_buf(self: *TrainState, batch_idx: usize, tensor: Tensor) void {
        std.debug.assert(self.donatable_count + batch_idx < self.input_bufs.len);
        self.input_bufs[self.donatable_count + batch_idx] = tensor.buffer() catch
            @panic("set_batch_buf: tensor must be device-backed");
    }

    /// Replace all non-donatable inputs from a flat tensor slice.
    ///
    /// `batch_tensors` must have exactly as many elements as there are
    ///  non-donatable inputs. All must be device-backed.
    pub fn set_batch(self: *TrainState, batch_tensors: []const Tensor) void {
        const batch_bufs = self.input_bufs[self.donatable_count..];
        std.debug.assert(batch_tensors.len == batch_bufs.len);
        for (batch_bufs, batch_tensors) |*slot, t| {
            slot.* = t.buffer() catch
                @panic("set_batch: tensor must be device-backed");
        }
    }

    pub const DeinitScope = enum {
        /// Free donatable (owned parameter) buffers only.
        donatable,
        /// Free non-donatable (borrowed batch) buffers only.
        non_donatable,
        /// Free all input buffers regardless of donation status.
        all,
    };

    /// Release device buffers and free internal allocations.
    ///
    /// Buffer release follows `scope`: `.donatable` frees only owned parameter
    ///  buffers, `.non_donatable` frees only borrowed batch buffers, `.all` frees
    ///  both. Internal slice allocations (input_bufs, output_bufs) are always
    ///  freed regardless of scope.
    ///
    /// After `deinit`, the TrainState must not be used.
    pub fn deinit(self: *TrainState, scope: DeinitScope) void {
        switch (scope) {
            .donatable => for (self.input_bufs[0..self.donatable_count]) |buf| self.backend.deinit_buffer(buf),
            .non_donatable => for (self.input_bufs[self.donatable_count..]) |buf| self.backend.deinit_buffer(buf),
            .all => for (self.input_bufs) |buf| self.backend.deinit_buffer(buf),
        }
        self.allocator.free(self.input_bufs);
        self.allocator.free(self.output_bufs);
    }

    /// Release device buffers without freeing internal allocations.
    ///
    /// Unlike `deinit`, the TrainState remains valid after this call. Only the
    ///  specified buffers are released (handles set to undefined). This enables
    ///  sequential calls: `release_buffers(.donatable)` then `release_buffers(.non_donatable)`.
    /// Follow with `deinit(.all)` to free internal slice allocations (no buffers
    ///  will be released since handles are already invalidated).
    pub fn release_buffers(self: *TrainState, scope: DeinitScope) void {
        const range = switch (scope) {
            .donatable => self.input_bufs[0..self.donatable_count],
            .non_donatable => self.input_bufs[self.donatable_count..],
            .all => self.input_bufs,
        };
        for (range) |*buf| {
            self.backend.deinit_buffer(buf.*);
            buf.* = undefined;
        }
    }
};

/// Compute non-donatable input indices from a spec tuple type and donated
///  argument positions.
///
/// `SpecsTuple` is the type of the spec tuple passed to `zg.trace` (e.g.
///  `@TypeOf(.{ params_spec, batch_spec })`). `donated` lists the argument
///  positions that are donatable (e.g. `&.{0}` means "first arg is donated").
///
/// Returns a comptime slice of i64 indices suitable for passing to
///  `TrainState.init` via `opts.non_donatable_input_indices`.
///
/// ```zig
/// const Specs = @TypeOf(.{ params_spec, batch_spec });
/// var state = try TrainState.init(allocator, exe, backend, tensors, output_arity, .{
///     .non_donatable_input_indices = comptime donate_argnums(Specs, &.{0}),
/// });
/// ```
pub fn donate_argnums(comptime SpecsTuple: type, comptime donated: []const usize) []const i64 {
    comptime {
        const info = @typeInfo(SpecsTuple);
        if (info != .@"struct" or !info.@"struct".is_tuple)
            @compileError("donate_argnums: SpecsTuple must be a tuple type");

        const fields = info.@"struct".fields;

        // Count non-donated leaves.
        var count: usize = 0;
        for (fields, 0..) |field, arg_idx| {
            if (!is_donated(donated, arg_idx)) {
                count += TensorTree.leaf_count(field.type);
            }
        }

        // Collect non-donated flat indices.
        var result: [count]i64 = undefined;
        var out_idx: usize = 0;
        var flat_offset: usize = 0;
        for (fields, 0..) |field, arg_idx| {
            const n = TensorTree.leaf_count(field.type);
            if (!is_donated(donated, arg_idx)) {
                for (0..n) |i| {
                    result[out_idx] = @intCast(flat_offset + i);
                    out_idx += 1;
                }
            }
            flat_offset += n;
        }

        const final = result;
        return &final;
    }
}

fn is_donated(comptime donated: []const usize, comptime arg_idx: usize) bool {
    for (donated) |d| {
        if (d == arg_idx) return true;
    }
    return false;
}
