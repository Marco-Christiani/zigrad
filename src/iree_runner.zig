/// Minimal IREE VMFB runner.
///
/// Loads a pre-compiled VMFB and executes a function with user-specified inputs.
/// This binary does NOT depend on zigrad, MLIR, PJRT, or TVM.
/// It links only the IREE runtime static archives + libc.
///
/// Usage:
///   iree-runner <vmfb-path> [options] [--input=<spec>...]
///
/// Options:
///   --function=<name>   Entry function (default: "module.main")
///   --driver=<name>     HAL driver (default: "local-sync")
///   --input=<spec>      Input tensor spec (repeatable). Format:
///                         <dim0>x<dim1>x...x<dtype>[=v0,v1,v2,...]
///                       If values are omitted, zeros are used.
///                       Supported dtypes: f32, f64, i32, i64
///
/// Examples:
///   iree-runner demo.vmfb --input=2x3xf32=1,2,3,4,5,6 --input=3x2xf32=7,8,9,10,11,12
///   iree-runner demo.vmfb  (no inputs, for zero-arg functions)
const std = @import("std");
const rt = @import("c/iree/runtime.zig");

const log = std.log.scoped(.@"zg/iree_runner");

pub fn main() !void {
    const gpa = std.heap.smp_allocator;

    var args = try std.process.argsWithAllocator(gpa);
    defer args.deinit();
    _ = args.next(); // skip argv[0]

    var vmfb_path: ?[]const u8 = null;
    var function_name: []const u8 = "module.main";
    var driver: []const u8 = "local-sync";
    var input_specs = std.ArrayList([]const u8).empty;
    defer input_specs.deinit(gpa);

    while (args.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--function=")) {
            function_name = arg["--function=".len..];
        } else if (std.mem.startsWith(u8, arg, "--driver=")) {
            driver = arg["--driver=".len..];
        } else if (std.mem.startsWith(u8, arg, "--input=")) {
            try input_specs.append(gpa, arg["--input=".len..]);
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("unknown option: {s}\n", .{arg});
            return error.UnknownOption;
        } else {
            vmfb_path = arg;
        }
    }

    const path = vmfb_path orelse {
        std.debug.print(
            \\usage: iree-runner <vmfb-path> [--function=module.main] [--driver=local-sync] [--input=<spec>...]
            \\
            \\  Input spec format: <dim0>x<dim1>x...x<dtype>[=v0,v1,...]
            \\  Supported dtypes: f32, f64, i32, i64
            \\
            \\  Example: iree-runner demo.vmfb --input=2x3xf32=1,2,3,4,5,6 --input=2x2xf32=2,2,2,2
            \\
        , .{});
        return error.MissingArgument;
    };

    // Read VMFB.
    const vmfb = blk: {
        var file = if (std.fs.path.isAbsolute(path))
            try std.fs.openFileAbsolute(path, .{})
        else
            try std.fs.cwd().openFile(path, .{});
        defer file.close();
        break :blk try file.readToEndAlloc(gpa, 256 * 1024 * 1024);
    };
    defer gpa.free(vmfb);

    log.info("loaded {d} bytes from {s}", .{ vmfb.len, path });

    // Create IREE runtime.
    const instance = try rt.instance_create();
    defer rt.instance_release(instance);

    const device = try rt.create_default_device(instance, driver);
    defer rt.device_release(device);

    const session = try rt.session_create(instance, device);
    defer rt.session_release(session);
    try rt.session_append_module(session, vmfb);

    const function = try rt.session_lookup_function(session, function_name);

    // Parse and create input buffers.
    var input_views = std.ArrayList(*rt.HalBufferView).empty;
    defer {
        for (input_views.items) |v| rt.buffer_view_release(v);
        input_views.deinit(gpa);
    }

    for (input_specs.items) |spec| {
        const view = try parse_and_create_input(gpa, device, spec);
        try input_views.append(gpa, view);
    }

    // Execute.
    var call = try rt.call_init(session, function);
    defer rt.call_deinit(&call);
    log.info("executed", .{});

    for (input_views.items) |view| {
        try rt.list_push_buffer_view(rt.call_inputs(&call), view);
    }

    try rt.call_invoke(&call);

    // Print outputs.
    const out_list = rt.call_outputs(&call);
    const n_out = rt.list_size(out_list);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch @panic("Flush failed");

    for (0..n_out) |i| {
        const out_view = try rt.list_get_buffer_view(out_list, i);
        defer rt.buffer_view_release(out_view);

        const elem_type = rt.buffer_view_element_type(out_view);
        const n_elem = rt.buffer_view_element_count(out_view);
        const byte_width = rt.element_byte_width(elem_type);
        const byte_count = n_elem * byte_width;

        const buf = try gpa.alloc(u8, byte_count);
        defer gpa.free(buf);
        try rt.buffer_view_to_host(out_view, buf);

        try stdout.print("output[{d}]: [", .{i});
        try print_typed_values(stdout, elem_type, buf, byte_count);
        try stdout.writeAll("]\n");
    }

    if (n_out == 0) {
        try stdout.writeAll("(no outputs)\n");
    }
}

// ---------------------------------------------------------------------------
// Element-type dispatch helpers.
// ---------------------------------------------------------------------------

/// Supported element types for printing and parsing, mapped to Zig types.
const element_type_map = .{
    .{ rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_32, f32 },
    .{ rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_64, f64 },
    .{ rt.c.IREE_HAL_ELEMENT_TYPE_SINT_32, i32 },
    .{ rt.c.IREE_HAL_ELEMENT_TYPE_SINT_64, i64 },
};

fn print_typed_values(writer: anytype, elem_type: rt.HalElementType, buf: []const u8, byte_count: usize) !void {
    inline for (element_type_map) |entry| {
        if (elem_type == entry[0]) {
            const T = entry[1];
            const vals: []const T = @alignCast(std.mem.bytesAsSlice(T, buf));
            for (vals, 0..) |v, j| {
                if (j > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{v});
            }
            return;
        }
    }
    try writer.print("<{d} bytes, unsupported element type>", .{byte_count});
}

fn parse_typed_value(elem_type: rt.HalElementType, trimmed: []const u8, data: []u8, idx: usize) !void {
    inline for (element_type_map) |entry| {
        if (elem_type == entry[0]) {
            const T = entry[1];
            const v = parse_scalar(T, trimmed) catch {
                log.err("invalid {s} value: '{s}'", .{ @typeName(T), trimmed });
                return error.InvalidValue;
            };
            @as(*align(1) T, @ptrCast(data.ptr + idx * @sizeOf(T))).* = v;
            return;
        }
    }
}

fn parse_scalar(comptime T: type, s: []const u8) !T {
    return switch (@typeInfo(T)) {
        .float => std.fmt.parseFloat(T, s),
        .int => std.fmt.parseInt(T, s, 10),
        else => @compileError("unsupported scalar type"),
    };
}

// ---------------------------------------------------------------------------
// Input spec parsing: "<dim0>x<dim1>x...x<dtype>[=v0,v1,...]"
// ---------------------------------------------------------------------------

const DTypeInfo = struct {
    element_type: rt.HalElementType,
    byte_width: usize,
};

fn parse_dtype(s: []const u8) ?DTypeInfo {
    const map = std.StaticStringMap(DTypeInfo).initComptime(.{
        .{ "f32", DTypeInfo{ .element_type = rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_32, .byte_width = 4 } },
        .{ "f64", DTypeInfo{ .element_type = rt.c.IREE_HAL_ELEMENT_TYPE_FLOAT_64, .byte_width = 8 } },
        .{ "i32", DTypeInfo{ .element_type = rt.c.IREE_HAL_ELEMENT_TYPE_SINT_32, .byte_width = 4 } },
        .{ "i64", DTypeInfo{ .element_type = rt.c.IREE_HAL_ELEMENT_TYPE_SINT_64, .byte_width = 8 } },
    });
    return map.get(s);
}

fn parse_and_create_input(
    gpa: std.mem.Allocator,
    device: *rt.HalDevice,
    spec: []const u8,
) !*rt.HalBufferView {
    // Split on '=' to separate shape+dtype from optional values.
    const eq_pos = std.mem.indexOfScalar(u8, spec, '=');
    const shape_dtype_str = if (eq_pos) |p| spec[0..p] else spec;
    const values_str: ?[]const u8 = if (eq_pos) |p| spec[p + 1 ..] else null;

    // Parse dimensions and dtype from "dim0xdim1x...xdtype".
    // The last 'x'-separated token that matches a dtype name is the dtype;
    // everything before it is dimensions.
    var parts = std.ArrayList([]const u8).empty;
    defer parts.deinit(gpa);

    var iter = std.mem.splitScalar(u8, shape_dtype_str, 'x');
    while (iter.next()) |part| {
        if (part.len > 0) try parts.append(gpa, part);
    }

    if (parts.items.len < 2) {
        log.err("invalid input spec (need at least one dim + dtype): {s}", .{spec});
        return error.InvalidInputSpec;
    }

    const dtype_str = parts.items[parts.items.len - 1];
    const dtype = parse_dtype(dtype_str) orelse {
        log.err("unsupported dtype '{s}' in input spec: {s}", .{ dtype_str, spec });
        return error.UnsupportedDtype;
    };

    const dim_parts = parts.items[0 .. parts.items.len - 1];

    var shape = try gpa.alloc(rt.HalDim, dim_parts.len);
    defer gpa.free(shape);

    var n_elem: usize = 1;
    for (dim_parts, 0..) |d, i| {
        shape[i] = std.fmt.parseInt(rt.HalDim, d, 10) catch {
            log.err("invalid dimension '{s}' in input spec: {s}", .{ d, spec });
            return error.InvalidDimension;
        };
        n_elem *= @intCast(shape[i]);
    }

    const byte_count = n_elem * dtype.byte_width;
    const data = try gpa.alloc(u8, byte_count);
    defer gpa.free(data);

    if (values_str) |vs| {
        // Parse comma-separated values.
        try parse_values(vs, dtype, data, n_elem);
    } else {
        @memset(data, 0);
    }

    return try rt.buffer_view_create_from_host(device, data, dtype.element_type, shape);
}

fn parse_values(vs: []const u8, dtype: DTypeInfo, data: []u8, n_elem: usize) !void {
    var val_iter = std.mem.splitScalar(u8, vs, ',');
    var idx: usize = 0;

    while (val_iter.next()) |v_str| {
        const trimmed = std.mem.trim(u8, v_str, " ");
        if (trimmed.len == 0) continue;
        if (idx >= n_elem) {
            log.err("too many values (expected {d})", .{n_elem});
            return error.TooManyValues;
        }

        try parse_typed_value(dtype.element_type, trimmed, data, idx);

        idx += 1;
    }

    if (idx < n_elem) {
        log.err("not enough values (got {d}, expected {d})", .{ idx, n_elem });
        return error.NotEnoughValues;
    }
}
