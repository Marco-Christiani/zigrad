/// IREE Backend
///
/// Implements the backend interface using IREE's two-phase model:
///   1. Compile: StableHLO MLIR -> VMFB (VM FlatBuffer) via `iree-compile`
///      (subprocess, default) or `libIREECompiler.so` (dlopen, when LLVM
///      versions align).
///   2. Execute: load VMFB into a runtime session and invoke via the IREE
///      runtime (statically linked).
///
/// CPU-only in the initial version (local-sync HAL driver).
///
/// ## Lifetime model
///
/// `Backend` owns the IREE runtime instance and the single HAL device.
/// `LoadedExecutable` owns the VMFB bytes and the session.
/// `Buffer` holds a retained `iree_hal_buffer_view_t*`; release via `deinit_buffer`.
/// `Event` is a no-op sentinel (local-sync is fully synchronous).
const std = @import("std");
const pr = @import("../pr/pr.zig");
const iree_compiler = @import("../c/iree/compiler.zig");
const rt = @import("../c/iree/runtime.zig");
const log = std.log.scoped(.@"zg/iree_backend");

/// Re-exported so callers can construct a Compiler without reaching into `c/iree/`.
pub const Compiler = iree_compiler.Compiler;

// ---------------------------------------------------------------------------
// Module-level associated types (required by interface.zig).
// ---------------------------------------------------------------------------

/// Wraps a retained `iree_hal_buffer_view_t*`.
pub const Buffer = struct {
    view: *rt.HalBufferView,
};

/// Non-owning view of a Buffer (same underlying pointer, no ownership).
pub const RawBuffer = struct {
    view: *rt.HalBufferView,
};

/// No-op event: local-sync execution is always complete by the time
/// the execute call returns.
pub const Event = struct {
    done: bool,
};

pub const Device = struct {
    device: *rt.HalDevice,
    id: usize,
};

/// Wraps a runtime session + resolved VM function + owned VMFB bytes.
///
/// The session borrows `vmfb` (no copy); `vmfb` must outlive the session.
/// `deinit_executable` releases the session then frees `vmfb`.
pub const LoadedExecutable = struct {
    session: *rt.Session,
    function: rt.VmFunction,
    /// VMFB bytes owned by this struct.  Freed after session_release.
    vmfb: []u8,
    allocator: std.mem.Allocator,
};

pub const ExecuteResult = struct {
    outputs: []Buffer,
    event: Event,
    allocator: std.mem.Allocator,
};

pub const CompileOptions = struct {
    /// Entry function name as it appears in the MLIR module (e.g. "main").
    /// IREE requires fully-qualified lookup ("module.<entry_name>").
    entry_name: []const u8 = "main",
};

/// Execute-time options (unused by IREE backend, present for interface conformance).
pub const ExecuteOptions = struct {};


// ---------------------------------------------------------------------------
// Backend struct.
// ---------------------------------------------------------------------------

/// IREE Backend owning the runtime instance and HAL device.
///
/// The compiler is optional -- pass `null` to `init` if you only need to
/// execute pre-compiled VMFB artifacts.
pub const Backend = struct {
    compiler: ?iree_compiler.Compiler,
    instance: *rt.Instance,
    hal_device: *rt.HalDevice,
    /// Wraps `hal_device` so it can be returned by `get_devices`.
    device_slot: Device,
    allocator: std.mem.Allocator,

    /// Initialize the backend.
    ///
    /// `compiler`: a pre-constructed `Compiler` (subprocess or dlopen mode), or
    ///  null to skip compilation (execute-only with pre-compiled VMFB artifacts).
    /// `driver_name`: HAL driver (default "local-sync").
    pub fn init(
        allocator: std.mem.Allocator,
        compiler: ?iree_compiler.Compiler,
        driver_name: []const u8,
    ) !Backend {
        errdefer if (compiler) |c| {
            var c_mut = c;
            c_mut.deinit();
        };

        // Create IREE runtime instance.
        const instance = try rt.instance_create();
        errdefer rt.instance_release(instance);

        // Create the default HAL device for the chosen driver.
        const hal_device = try rt.create_default_device(instance, driver_name);
        errdefer rt.device_release(hal_device);

        log.info("IREE backend: driver={s}", .{driver_name});

        const self: Backend = .{
            .compiler = compiler,
            .instance = instance,
            .hal_device = hal_device,
            .device_slot = .{ .device = hal_device, .id = 0 },
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *Backend) void {
        rt.device_release(self.hal_device);
        rt.instance_release(self.instance);
        if (self.compiler) |*c| c.deinit();
    }

    // -----------------------------------------------------------------------
    // Lifecycle.
    // -----------------------------------------------------------------------

    /// Return a single-element slice containing the CPU device.
    ///
    /// Caller owns the returned slice; free with `allocator.free`.
    pub fn get_devices(self: *Backend, allocator: std.mem.Allocator) ![]Device {
        const devs = try allocator.alloc(Device, 1);
        devs[0] = self.device_slot;
        return devs;
    }

    // -----------------------------------------------------------------------
    // Compilation.
    // -----------------------------------------------------------------------

    /// Compile MLIR bytes to a `LoadedExecutable`.
    ///
    /// 1. Invokes the IREE compiler to produce VMFB bytes.
    /// 2. Creates a runtime session, appends the VMFB module.
    /// 3. Looks up the entry function (`opts.entry_name`).
    ///
    /// Precondition: backend was initialized with a compiler.
    pub fn compile(
        self: *Backend,
        device: *const Device,
        mlir_bytes: []const u8,
        is_bytecode: bool,
        opts: CompileOptions,
    ) !LoadedExecutable {
        const cmp = self.compiler orelse {
            log.err("compile() called but no compiler library loaded", .{});
            return error.NoCompiler;
        };

        // Phase 1: MLIR -> VMFB.
        const vmfb = try cmp.compile(self.allocator, mlir_bytes, is_bytecode);
        errdefer self.allocator.free(vmfb);

        log.debug("VMFB size: {d} bytes", .{vmfb.len});

        // Phase 2: load VMFB into a runtime session.
        const session = try rt.session_create(self.instance, device.device);
        errdefer rt.session_release(session);

        // Session borrows vmfb bytes (null allocator = no copy).
        try rt.session_append_module(session, vmfb);

        // Phase 3: resolve the entry function.
        // IREE requires fully-qualified names: "module.<func_name>".
        var fq_buf: [256]u8 = undefined;
        const fq_name = std.fmt.bufPrint(&fq_buf, "module.{s}", .{opts.entry_name}) catch {
            log.err("entry name too long for IREE lookup: '{s}'", .{opts.entry_name});
            return error.EntryNameTooLong;
        };
        const function = try rt.session_lookup_function(session, fq_name);

        return .{
            .session = session,
            .function = function,
            .vmfb = vmfb,
            .allocator = self.allocator,
        };
    }

    // -----------------------------------------------------------------------
    // Buffer management.
    // -----------------------------------------------------------------------

    /// Upload `data` to a device buffer view.
    ///
    /// `shape` is in row-major order; elements are `dtype`.
    /// The returned buffer holds a retained reference.
    pub fn buffer_from_host(
        self: *Backend,
        device: *const Device,
        data: []const u8,
        dtype: pr.DType,
        shape: []const i64,
    ) !Buffer {
        _ = self;
        const elem_type = dtype_to_element_type(dtype);

        comptime std.debug.assert(@sizeOf(rt.HalDim) == @sizeOf(i64));
        const hal_shape: []const rt.HalDim = @ptrCast(shape);

        const view = try rt.buffer_view_create_from_host(
            device.device,
            data,
            elem_type,
            hal_shape,
        );
        return .{ .view = view };
    }

    // -----------------------------------------------------------------------
    // Execution.
    // -----------------------------------------------------------------------

    /// Execute `exe` with `inputs`, returning freshly allocated output buffers.
    ///
    /// The caller owns `result.outputs`; free each with `deinit_buffer` then
    /// free the slice with `allocator.free(result.outputs)`.
    pub fn execute(
        self: *Backend,
        exe: *LoadedExecutable,
        allocator: std.mem.Allocator,
        inputs: []const Buffer,
        _: ExecuteOptions,
    ) !ExecuteResult {
        var call = try rt.call_init(exe.session, exe.function);
        defer rt.call_deinit(&call);

        // Push inputs (each retains an extra ref; the call's list holds it).
        for (inputs) |buf| {
            try rt.list_push_buffer_view(rt.call_inputs(&call), buf.view);
        }

        try rt.call_invoke(&call);

        // Collect outputs.
        const out_list = rt.call_outputs(&call);
        const n_out = rt.list_size(out_list);
        const outputs = try allocator.alloc(Buffer, n_out);
        errdefer {
            for (outputs) |*b| self.deinit_buffer(b);
            allocator.free(outputs);
        }

        for (0..n_out) |i| {
            const view = try rt.list_get_buffer_view(out_list, i);
            outputs[i] = .{ .view = view };
        }

        return .{
            .outputs = outputs,
            .event = .{ .done = true },
            .allocator = allocator,
        };
    }

    /// Execute `exe` with `inputs`, writing results into pre-allocated `outputs`.
    ///
    /// For the synchronous CPU backend this always completes immediately and
    /// returns `null` (no event to await).
    ///
    /// Outputs are written by copying IREE's results through host memory into
    /// the pre-allocated buffer views.  This involves extra copies for the
    /// CPU path but is correct.
    pub fn execute_into(
        self: *Backend,
        exe: *LoadedExecutable,
        inputs: []const RawBuffer,
        outputs: []?RawBuffer,
        non_donatable: ?[]const i64,
        _: ExecuteOptions,
    ) !?Event {
        _ = non_donatable;

        var call = try rt.call_init(exe.session, exe.function);
        defer rt.call_deinit(&call);

        for (inputs) |raw| {
            try rt.list_push_buffer_view(rt.call_inputs(&call), raw.view);
        }

        try rt.call_invoke(&call);

        // Copy each output into the pre-allocated destination buffer.
        const out_list = rt.call_outputs(&call);
        for (outputs, 0..) |maybe_dst, i| {
            const src_view = try rt.list_get_buffer_view(out_list, i);
            defer rt.buffer_view_release(src_view);

            if (maybe_dst) |dst| {
                try copy_buffer_view(self.allocator, src_view, dst.view);
            }
        }

        return null;
    }

    // -----------------------------------------------------------------------
    // Handle lifecycle.
    // -----------------------------------------------------------------------

    pub fn deinit_buffer(self: *Backend, buf: *Buffer) void {
        _ = self;
        rt.buffer_view_release(buf.view);
        buf.* = undefined;
    }

    pub fn deinit_event(self: *Backend, event: *Event) void {
        _ = self;
        _ = event;
        // No-op: local-sync backend has no async events.
    }

    pub fn deinit_executable(self: *Backend, exe: *LoadedExecutable) void {
        _ = self;
        // Release session first (it borrows vmfb), then free vmfb.
        rt.session_release(exe.session);
        exe.allocator.free(exe.vmfb);
        exe.* = undefined;
    }

    // -----------------------------------------------------------------------
    // Data transfer.
    // -----------------------------------------------------------------------

    /// Copy the buffer view contents to `dst` on the host.
    ///
    /// Returns immediately (local-sync is synchronous).
    pub fn buffer_to_host(self: *Backend, buf: *Buffer, dst: []u8) !Event {
        _ = self;
        try rt.buffer_view_to_host(buf.view, dst);
        return .{ .done = true };
    }

    /// Block until the event is complete.
    ///
    /// No-op for the synchronous CPU backend.
    pub fn await_event(self: *Backend, event: *Event) void {
        _ = self;
        _ = event;
    }
};

// ---------------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------------

/// Map a PR DType to the IREE HAL element type constant.
fn dtype_to_element_type(dtype: pr.DType) rt.HalElementType {
    return switch (dtype) {
        .f32 => rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        .f64 => rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_64,
        .bf16 => rt.c.IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
        .i32 => rt.c.IREE_HAL_ELEMENT_TYPE_SINT_32,
        .i64 => rt.c.IREE_HAL_ELEMENT_TYPE_SINT_64,
        .u32 => rt.c.IREE_HAL_ELEMENT_TYPE_UINT_32,
        .u64 => rt.c.IREE_HAL_ELEMENT_TYPE_UINT_64,
        .bool => rt.c.IREE_HAL_ELEMENT_TYPE_BOOL_8,
    };
}

/// Copy the contents of `src` buffer view into `dst` buffer view via host memory.
fn copy_buffer_view(
    allocator: std.mem.Allocator,
    src: *rt.HalBufferView,
    dst: *rt.HalBufferView,
) !void {
    const n_elem = rt.buffer_view_element_count(src);
    const etype = rt.buffer_view_element_type(src);
    const byte_count = n_elem * rt.element_byte_width(etype);

    const tmp = try allocator.alloc(u8, byte_count);
    defer allocator.free(tmp);

    try rt.buffer_view_to_host(src, tmp);
    try rt.buffer_view_from_host(dst, tmp);
}

