/// StableHLO Lowering
///
/// Lowers PR (Program Representation) to StableHLO MLIR.
/// This is the core PR -> MLIR boundary in the pass-based pipeline.
///
/// Provides:
/// - lower_pass: Pass function for pipeline integration
/// - lower_function_to_mlir: Direct lowering API
///
/// See KB: "Pass-Based Pipeline Direction (Design Update)"
const std = @import("std");

const pr = @import("../pr/pr.zig");
const ops = @import("../pr/ops/ops.zig");
const mlir = @import("../ffi/mlir/mlir.zig");
const pass = @import("../pipeline/pass.zig");
const log = std.log.scoped(.@"zg/lower_stablehlo");

pub const LowerError = ops.types.LowerError;

pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

// ============================================================================
// Core Lowering Implementation
// ============================================================================

/// Lower a PR program to StableHLO MLIR bytecode or text.
///
/// Entry Function Selection and Naming:
/// - `entry_name`: Selects which PR function is the compilation entry point.
///   - If provided: That function is renamed to "@main" in MLIR.
///   - If null: "main" (or the only function) is used as entry.
/// - All PR functions are lowered into the MLIR module to preserve whole-program
///   optimization opportunities unless explicit boundaries exist.
/// - The entry function is always renamed to "@main" in the MLIR output. This is
///   required by XLA/PJRT.
/// - Non-entry functions retain their PR names, except when a non-entry function
///   is already named "main" and a different entry is selected; that symbol is
///   renamed to avoid collisions.
///
/// Example:
///   Program with functions ["forward", "backward"]
///   - entry_name = "backward" → MLIR contains only @main (was "backward")
///   - entry_name = null → MLIR contains @main (was "forward") + @backward
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    out: OutputFormat,
) ![]u8 {
    pr.validate_program(program) catch return error.InvalidProgram;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var registry = try mlir.Registry.init();
    defer registry.deinit();

    mlir.DialectHandle.from_string("func").insert_dialect(registry);
    mlir.DialectHandle.from_string("stablehlo").insert_dialect(registry);

    var ctx = try mlir.Context.init_with_registry(registry, false);
    defer ctx.deinit();
    ctx.allow_unregistered_dialects(false);

    const func_handle = mlir.DialectHandle.from_string("func");
    func_handle.register_dialect(ctx);
    _ = func_handle.load_dialect(ctx);

    const stablehlo_handle = mlir.DialectHandle.from_string("stablehlo");
    stablehlo_handle.register_dialect(ctx);
    _ = stablehlo_handle.load_dialect(ctx);

    const loc = mlir.Location.unknown(ctx);

    var module = mlir.Module.init(loc);
    defer module.deinit();

    const entry_index = try find_entry_function(program, entry_name);

    // XLA/PJRT requires the entry function to be named "main".
    for (program.functions, 0..) |func, idx| {
        const sym_name = try choose_symbol_name(arena, program, idx, entry_index, entry_name);
        try lower_function_into_module(arena, ctx, module, func, sym_name);
    }

    if (!module.op().verify()) return error.InvalidMlir;

    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (out) {
        .mlir_bytecode => try module.op().write_bytecode(&writer_state.writer),
        .mlir_text => try module.op().print(&writer_state.writer, .{}),
    }

    return try writer_state.toOwnedSlice();
}

pub fn lower_function_to_mlir(allocator: std.mem.Allocator, func: pr.Function, out: OutputFormat) ![]u8 {
    var program = pr.Program.init(allocator);
    defer program.deinit();

    try program.add_function(func);
    return lower_program_to_mlir(allocator, &program, func.name, out);
}

fn should_outline_eqn(ctx: ops.types.LowerContext, eqn: pr.Eqn) bool {
    const params = ctx.params(eqn);
    if (pr.param_outline(params) orelse false) return true;
    if (pr.param_kernelize_provider(params) != null) return true;
    return false;
}

fn lower_outlined_eqn(
    arena: std.mem.Allocator,
    outlined_index: *usize,
    outlined_prefix: []const u8,
    mlir_ctx: mlir.Context,
    module: mlir.Module,
    ctx: ops.types.LowerContext,
    eqn: pr.Eqn,
) !void {
    const inputs = ctx.inputs(eqn);
    const outputs = ctx.outputs(eqn);
    const params = ctx.params(eqn);

    // v0: all current PR ops are single-output; keep outlining strict.
    if (outputs.len != 1) return error.InvalidProgram;
    if (inputs.len == 0) return error.InvalidProgram;

    const out_id = outputs[0];
    const out_tensor = try ctx.tensor_of(out_id);
    const out_type = try ctx.tensor_to_mlir_type(out_tensor);

    const callee_name = try std.fmt.allocPrint(arena, "{s}_outlined_{d}", .{ outlined_prefix, outlined_index.* });
    outlined_index.* += 1;
    const callee_name_z = try arena.allocSentinel(u8, callee_name.len, 0);
    @memcpy(callee_name_z, callee_name);

    // Build callee signature (inputs -> output).
    const callee_param_types = try arena.alloc(mlir.Type, inputs.len);
    const callee_param_locs = try arena.alloc(mlir.Location, inputs.len);
    for (inputs, 0..) |in_id, i| {
        const in_tensor = try ctx.tensor_of(in_id);
        callee_param_types[i] = try ctx.tensor_to_mlir_type(in_tensor);
        callee_param_locs[i] = ctx.loc;
    }
    const callee_result_types = &[_]mlir.Type{out_type};
    const callee_fn_type = mlir.Type.function(mlir_ctx, callee_param_types, callee_result_types);

    // Build callee body.
    const callee_entry = try mlir.Block.init(callee_param_types, callee_param_locs);

    const callee_value_map = try arena.alloc(?mlir.Value, ctx.func.avals.len);
    @memset(callee_value_map, null);
    for (inputs, 0..) |in_id, i| callee_value_map[@intCast(in_id)] = callee_entry.argument(i);

    const callee_ctx = ops.types.LowerContext{
        .mlir_ctx = mlir_ctx,
        .block = callee_entry,
        .loc = ctx.loc,
        .value_map = callee_value_map,
        .func = ctx.func,
        .arena = arena,
    };

    try ops.lower(callee_ctx, eqn);

    const callee_out = callee_value_map[@intCast(out_id)] orelse return error.InvalidProgram;
    const callee_ret = mlir.Operation.make(mlir_ctx, "func.return", .{
        .operands = &.{callee_out},
        .verify = false,
        .location = ctx.loc,
    });
    callee_entry.append_operation(callee_ret);

    const callee_op = mlir.Operation.make(mlir_ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{callee_entry},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(mlir_ctx, callee_name) },
            .{ "function_type", mlir.Attribute.type_(callee_fn_type) },
            // Best-effort: discourage inlining to preserve a visible boundary.
            .{ "llvm.noinline", mlir.Attribute.unit(mlir_ctx) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    if (pr.param_kernelize_provider(params)) |provider| {
        // Tag the outlined function so later pipeline/toolchain stages can identify and
        // replace/compile it via the selected kernelization provider.
        callee_op.set_attribute_by_name("zigrad.kernelize.provider", mlir.Attribute.string(mlir_ctx, provider));
    }
    module.get_body().append_operation(callee_op);

    // Emit a call in the original block.
    const call_operands = try arena.alloc(mlir.Value, inputs.len);
    for (inputs, 0..) |in_id, i| call_operands[i] = ctx.get_value(in_id) orelse return error.InvalidProgram;

    const call_op = mlir.Operation.make(mlir_ctx, "func.call", .{
        .results = &.{out_type},
        .operands = call_operands,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(mlir_ctx, callee_name_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });
    ctx.block.append_operation(call_op);
    ctx.set_value(out_id, call_op.result(0));
}

fn lower_function_into_module(
    arena: std.mem.Allocator,
    ctx: mlir.Context,
    module: mlir.Module,
    func: pr.Function,
    sym_name: []const u8,
) !void {
    const loc = mlir.Location.unknown(ctx);

    const param_types = try arena.alloc(mlir.Type, func.params.len);
    const param_locs = try arena.alloc(mlir.Location, func.params.len);
    for (func.params, 0..) |param_id, i| {
        const tensor = func.avals[@intCast(param_id)].as_tensor() orelse return error.InvalidProgram;
        param_types[i] = try tensor_to_mlir_type(ctx, tensor, arena);
        param_locs[i] = loc;
    }

    const result_types = try arena.alloc(mlir.Type, func.returns.len);
    for (func.returns, 0..) |ret_id, i| {
        const tensor = func.avals[@intCast(ret_id)].as_tensor() orelse return error.InvalidProgram;
        result_types[i] = try tensor_to_mlir_type(ctx, tensor, arena);
    }

    const fn_type = mlir.Type.function(ctx, param_types, result_types);

    const entry_block = try mlir.Block.init(param_types, param_locs);

    const value_map = try arena.alloc(?mlir.Value, func.avals.len);
    @memset(value_map, null);
    for (func.params, 0..) |param_id, i| {
        value_map[@intCast(param_id)] = entry_block.argument(i);
    }

    const lower_ctx = ops.types.LowerContext{
        .mlir_ctx = ctx,
        .block = entry_block,
        .loc = loc,
        .value_map = value_map,
        .func = func,
        .arena = arena,
    };

    var outlined_index: usize = 0;
    const outlined_prefix = if (sym_name.len == 0) "func" else sym_name;
    for (func.eqns) |eqn| {
        if (should_outline_eqn(lower_ctx, eqn)) {
            try lower_outlined_eqn(arena, &outlined_index, outlined_prefix, ctx, module, lower_ctx, eqn);
        } else {
            try ops.lower(lower_ctx, eqn);
        }
    }

    const ret_values = try arena.alloc(mlir.Value, func.returns.len);
    for (func.returns, 0..) |ret_id, i| {
        ret_values[i] = value_map[@intCast(ret_id)] orelse return error.InvalidProgram;
    }

    const return_op = mlir.Operation.make(ctx, "func.return", .{
        .operands = ret_values,
        .verify = false,
        .location = loc,
    });
    entry_block.append_operation(return_op);

    const func_op = mlir.Operation.make(ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(ctx, sym_name) },
            .{ "function_type", mlir.Attribute.type_(fn_type) },
        },
        .verify = false,
        .location = loc,
    });
    module.get_body().append_operation(func_op);
}

fn find_entry_function(program: *const pr.Program, entry_name: ?[]const u8) !usize {
    if (program.functions.len == 0) return error.InvalidProgram;

    if (entry_name) |name| {
        for (program.functions, 0..) |func, idx| {
            if (std.mem.eql(u8, func.name, name)) return idx;
        }
        return error.InvalidProgram;
    }

    for (program.functions, 0..) |func, idx| {
        if (std.mem.eql(u8, func.name, "main")) return idx;
    }

    if (program.functions.len == 1) return 0;
    return error.InvalidProgram;
}

fn choose_symbol_name(
    arena: std.mem.Allocator,
    program: *const pr.Program,
    idx: usize,
    entry_index: usize,
    entry_name: ?[]const u8,
) ![]const u8 {
    if (idx == entry_index) return "main";

    const func = program.functions[idx];
    if (entry_name == null or !std.mem.eql(u8, func.name, "main")) return func.name;

    var suffix: usize = 0;
    while (true) : (suffix += 1) {
        const candidate = try std.fmt.allocPrint(arena, "main_non_entry_{d}", .{suffix});
        if (!is_symbol_name_used(program, entry_index, candidate)) {
            log.warn("renaming non-entry function 'main' to '{s}' to avoid entry collision", .{candidate});
            return candidate;
        }
    }
}

fn is_symbol_name_used(program: *const pr.Program, entry_index: usize, name: []const u8) bool {
    for (program.functions, 0..) |func, idx| {
        if (idx == entry_index) continue;
        if (std.mem.eql(u8, func.name, name)) return true;
    }
    return false;
}

fn tensor_to_mlir_type(ctx: mlir.Context, t: pr.Tensor, arena: std.mem.Allocator) !mlir.Type {
    const dims_i64 = try arena.alloc(i64, t.shape.dims.len);
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, ops.types.dtype_to_mlir_type(ctx, t.dtype));
}

// ============================================================================
// Pass Integration
// ============================================================================

/// Lower pass: PR artifact -> MLIR artifact.
pub const LowerPassConfig = struct {
    encoding: pass.MlirEncoding = .bytecode,

    /// Selects which PR function is the compilation entry point.
    /// The selected function is always renamed to "@main" in MLIR output (XLA requirement).
    entry_name: ?[]const u8 = null,
};

pub fn lower_pass(artifact: *pass.Artifact, ctx: *pass.PassContext, userdata: ?*anyopaque) pass.PassError!void {
    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const cfg_ptr = userdata orelse return error.MissingContext;
    const cfg: *LowerPassConfig = @ptrCast(@alignCast(cfg_ptr));

    const program = artifact.pr;

    const mlir_bytes = switch (cfg.encoding) {
        .text => lower_program_to_mlir(ctx.allocator, program, cfg.entry_name, .mlir_text) catch return error.LoweringFailed,
        .bytecode => lower_program_to_mlir(ctx.allocator, program, cfg.entry_name, .mlir_bytecode) catch return error.LoweringFailed,
    };

    artifact.replace(ctx.allocator, .{
        .mlir = .{
            .bytes = mlir_bytes,
            .encoding = cfg.encoding,
        },
    });
}

/// Metadata for the lower pass.
pub fn lower_pass_with_config(config: *LowerPassConfig) pass.Pass {
    return .{
        .name = "stablehlo_lower",
        .input_kind = .pr,
        .output_kind = .mlir,
        .run = lower_pass,
        .userdata = config,
    };
}

/// Validate pass: PR artifact -> PR artifact.
fn validate_pass_run(artifact: *pass.Artifact, ctx: *pass.PassContext, _: ?*anyopaque) pass.PassError!void {
    _ = ctx;

    if (artifact.kind() != .pr) return error.ArtifactKindMismatch;

    const program = artifact.pr;
    pr.validate_program(program) catch return error.ValidationFailed;
}

/// Metadata for the validate pass.
pub const validate_pass = pass.Pass{
    .name = "pr_validate",
    .input_kind = .pr,
    .output_kind = .pr,
    .run = validate_pass_run,
};

/// Convenience: lower with encoding preference.
pub fn lower(
    allocator: std.mem.Allocator,
    program: *const pr.Program,
    entry_name: ?[]const u8,
    encoding: pass.MlirEncoding,
) !pass.MlirArtifact {
    const format: OutputFormat = switch (encoding) {
        .text => .mlir_text,
        .bytecode => .mlir_bytecode,
    };

    const bytes = try lower_program_to_mlir(allocator, program, entry_name, format);

    return .{
        .bytes = bytes,
        .encoding = encoding,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "lowering produces verified bytecode" {
    var program = try @import("../frontend/frontend.zig").build_demo_program(std.testing.allocator);
    defer program.deinit();

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering keeps non-entry functions when entry_name is set" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var fwd_builder = try pr.FunctionBuilder.init(&program, "forward");
    defer fwd_builder.deinit();
    const fwd_x = try fwd_builder.param_tensor(.f32, &.{ 2 });
    const fwd = try fwd_builder.finish(&.{fwd_x});
    try program.add_function(fwd);

    var bwd_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer bwd_builder.deinit();
    const bwd_x = try bwd_builder.param_tensor(.f32, &.{ 2 });
    const bwd = try bwd_builder.finish(&.{bwd_x});
    try program.add_function(bwd);

    const text = try lower_program_to_mlir(testing.allocator, &program, "backward", .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @forward") != null);
}

test "lowering renames non-entry main when entry_name differs" {
    const testing = std.testing;

    var program = pr.Program.init(testing.allocator);
    defer program.deinit();

    var main_builder = try pr.FunctionBuilder.init(&program, "main");
    defer main_builder.deinit();
    const main_x = try main_builder.param_tensor(.f32, &.{ 2 });
    const main_fn = try main_builder.finish(&.{main_x});
    try program.add_function(main_fn);

    var other_builder = try pr.FunctionBuilder.init(&program, "backward");
    defer other_builder.deinit();
    const other_x = try other_builder.param_tensor(.f32, &.{ 2 });
    const other_fn = try other_builder.finish(&.{other_x});
    try program.add_function(other_fn);

    const text = try lower_program_to_mlir(testing.allocator, &program, "backward", .mlir_text);
    defer testing.allocator.free(text);

    try testing.expect(std.mem.indexOf(u8, text, "func.func @main") != null);
    try testing.expect(std.mem.indexOf(u8, text, "func.func @main_non_entry_0") != null);
}

test "lowering supports reshape/broadcast/transpose" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const t = try b.transpose(x, &.{ 1, 0 });
    const r = try b.reshape(t, &.{6});
    const y = try b.broadcast_in_dim(r, &.{ 2, 6 }, &.{1});

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports custom_call boundary" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.custom_call("zigrad.test.missing_handler", &.{x}, x);

    const func = try b.finish(&.{y});
    try program.add_function(func);

    const bc = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering supports vjp matmul demo" {
    var program = try @import("../frontend/frontend.zig").build_demo_program(std.testing.allocator);
    defer program.deinit();

    const fwd = program.functions[0];
    const vjp_func = try @import("../pr/ad.zig").vjp(std.testing.allocator, &program, fwd, "vjp");

    try program.add_function(vjp_func);
    const bc = try lower_program_to_mlir(std.testing.allocator, &program, "vjp", .mlir_bytecode);
    defer std.testing.allocator.free(bc);
    try std.testing.expect(bc.len > 0);
}

test "lowering can outline an equation into a call boundary" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.emit(.dot, &.{ a, c }, &.{.{ .outline = true }});
    const func = try b.finish(&.{d});
    try program.add_function(func);

    const text = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "func.call") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "main_outlined_0") != null);
}

test "lowering tags kernelize provider on outlined functions" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a = try b.param_tensor(.f32, &.{ 2, 3 });
    const c = try b.param_tensor(.f32, &.{ 3, 2 });
    const d = try b.emit(.dot, &.{ a, c }, &.{.{ .kernelize_provider = "tvm" }});
    const func = try b.finish(&.{d});
    try program.add_function(func);

    const text = try lower_program_to_mlir(std.testing.allocator, &program, null, .mlir_text);
    defer std.testing.allocator.free(text);

    try std.testing.expect(std.mem.indexOf(u8, text, "zigrad.kernelize.provider") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "tvm") != null);
}

test "lower pass produces MLIR artifact" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try pr.FunctionBuilder.init(&program, "test");
    defer b.deinit();
    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.param_tensor(.f32, &.{ 3, 2 });
    const z = try b.dot(x, y);
    const func = try b.finish(&.{z});

    var ctx = pass.PassContext{
        .allocator = std.testing.allocator,
    };
    var cfg = LowerPassConfig{ .encoding = .bytecode };

    try program.add_function(func);
    var output = pass.Artifact{ .pr = &program };
    try lower_pass(&output, &ctx, &cfg);
    defer output.deinit(std.testing.allocator);

    try std.testing.expectEqual(pass.ArtifactKind.mlir, output.kind());
    try std.testing.expect(output.mlir.bytes.len > 0);
}
