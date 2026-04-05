const std = @import("std");

const backend = @import("../backend/root.zig");
const Backend = backend.Backend;
const Tensor = @import("../tensor.zig");
const frontend = @import("frontend.zig");

/// Manages device buffer state across training steps.
///
/// `TrainState` is the execute-side counterpart to `frontend.compile`. After
///  compiling a train step (typically a function that uses
///  `transforms.value_and_grad` + an optimizer), `TrainState` handles the
///  repetitive buffer plumbing:
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
/// var compiled = try frontend.compile(train_step, alloc, b, device, specs, cfg);
/// var state = try TrainState.init_from_model(alloc, &compiled, b, initial_tensors, .{});
/// for (0..num_steps) |_| {
///     state.set_batch(batch_tensors);
///     const result = try state.step();
///     const loss_val = try result.loss.item(f32);
///     result.loss.deinit();
/// }
/// state.deinit(.all);
/// ```
///
/// The donation mask comes from the `donatable` field on abstract Tensor specs
///  passed to `frontend.compile`. See `Tensor.abstract` and
///  `Tensor.AbstractOpts`.
pub const TrainState = struct {
    input_bufs: []Backend.Buffer,
    output_bufs: []Backend.Buffer,
    exe: Backend.Executable,
    backend: *Backend,
    /// Per-input donation mask from the compiled model.
    donatable: []const bool,
    non_donatable_indices: []const i64,
    /// DType of the loss output (output[0]). Used to wrap the raw buffer
    ///  as a Tensor in `StepResult`.
    loss_dtype: @import("../pr/pr.zig").DType,
    allocator: std.mem.Allocator,

    pub const StepResult = struct {
        loss: Tensor,
        event: ?Backend.Event,
    };

    /// Initialize from a CompiledModel and a flat tensor slice.
    ///
    /// `initial_tensors` must have exactly `compiled.input_arity` elements,
    ///  in the same flattened order as the spec tree passed to `compile`.
    ///  All tensors must be device-backed.
    /// Ownership follows the donation mask: donatable buffers are owned
    ///  by TrainState (freed on swap/deinit), non-donatable buffers are
    ///  borrowed (caller manages lifetime).
    pub const InitOpts = struct {
        /// DType of the loss output (output[0]). Defaults to f32.
        /// Override for mixed-precision training (e.g. bf16 loss with
        ///  f32 upcast).
        loss_dtype: @import("../pr/pr.zig").DType = .f32,
    };

    pub fn init_from_model(
        allocator: std.mem.Allocator,
        compiled: *frontend.CompiledModel,
        backend_: *Backend,
        initial_tensors: []const Tensor,
        opts: InitOpts,
    ) !TrainState {
        if (initial_tensors.len != compiled.input_arity) return error.InvalidInputCount;

        const input_bufs = try allocator.alloc(Backend.Buffer, compiled.input_arity);
        for (input_bufs, initial_tensors) |*slot, t| {
            slot.* = try t.buffer();
        }

        const output_bufs = try allocator.alloc(Backend.Buffer, compiled.output_arity);

        const non_donatable_indices = try compiled.non_donatable_indices(allocator);

        return .{
            .input_bufs = input_bufs,
            .output_bufs = output_bufs,
            .exe = compiled.exe,
            .backend = backend_,
            .donatable = compiled.donatable,
            .non_donatable_indices = non_donatable_indices,
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
        var out_idx: usize = 1;
        for (self.input_bufs, self.donatable) |*old, is_donatable| {
            if (is_donatable) {
                const new_buf = self.output_bufs[out_idx];
                if (new_buf.handle != old.handle) {
                    self.backend.deinit_buffer(old.*);
                    old.* = new_buf;
                }
                out_idx += 1;
            }
        }

        return .{
            .loss = Tensor.from_buffer(self.backend, loss_buf, self.loss_dtype, &.{}),
            .event = event,
        };
    }

    /// Replace a non-donatable (borrowed) input by index.
    ///
    /// The tensor must be device-backed. The old buffer is NOT freed
    ///  (caller owns non-donatable buffers).
    pub fn set_batch_buf(self: *TrainState, input_idx: usize, tensor: Tensor) void {
        std.debug.assert(!self.donatable[input_idx]);
        self.input_bufs[input_idx] = tensor.buffer() catch
            @panic("set_batch_buf: tensor must be device-backed");
    }

    /// Replace all non-donatable inputs from a flat tensor slice.
    ///
    /// `batch_tensors` must have exactly as many elements as there are
    ///   non-donatable inputs, in spec-tree order.
    /// All must be device-backed.
    pub fn set_batch(self: *TrainState, batch_tensors: []const Tensor) void {
        var batch_idx: usize = 0;
        for (self.input_bufs, self.donatable) |*slot, is_donatable| {
            if (!is_donatable) {
                slot.* = batch_tensors[batch_idx].buffer() catch
                    @panic("set_batch: tensor must be device-backed");
                batch_idx += 1;
            }
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
    ///  both. Internal slice allocations (input_bufs, output_bufs, non_donatable_indices)
    ///  are always freed regardless of scope.
    ///
    /// After `deinit`, the TrainState must not be used. To free donatable and
    ///  non-donatable buffers at different times, call `release_buffers` first
    ///  (which preserves internal allocations), then `deinit` with any scope to
    ///  clean up the rest.
    pub fn deinit(self: *TrainState, scope: DeinitScope) void {
        for (self.input_bufs, self.donatable) |buf, is_donatable| {
            switch (scope) {
                .donatable => if (is_donatable) self.backend.deinit_buffer(buf),
                .non_donatable => if (!is_donatable) self.backend.deinit_buffer(buf),
                .all => self.backend.deinit_buffer(buf),
            }
        }
        self.allocator.free(self.input_bufs);
        self.allocator.free(self.output_bufs);
        self.allocator.free(self.non_donatable_indices);
    }

    /// Release device buffers without freeing internal allocations.
    ///
    /// Unlike `deinit`, the TrainState remains valid after this call. Only the
    ///  specified buffers are released (handles set to undefined). This enables
    ///  sequential calls: `release_buffers(.donatable)` then `release_buffers(.non_donatable)`.
    /// Follow with `deinit(.all)` to free internal slice allocations (no buffers
    ///  will be released since handles are already invalidated).
    pub fn release_buffers(self: *TrainState, scope: DeinitScope) void {
        for (self.input_bufs, self.donatable) |*buf, is_donatable| {
            const should_release = switch (scope) {
                .donatable => is_donatable,
                .non_donatable => !is_donatable,
                .all => true,
            };
            if (should_release) {
                self.backend.deinit_buffer(buf.*);
                buf.* = undefined;
            }
        }
    }
};
