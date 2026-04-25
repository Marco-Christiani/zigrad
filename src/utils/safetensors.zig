//! Utils for populating Tensor containers from safetensors files.
const std = @import("std");
const stz = @import("safetensors_zg");
const Tensor = @import("../tensor.zig");
const DType = @import("../pr/pr.zig").DType;

pub const Opts = struct {
    allocator: std.mem.Allocator,
    dtype: DType,
};

/// Load a typed struct of `Tensor` leaves from a safetensors file.
///
/// The user declares a struct of type `T` whose field paths mirror
///  the safetensors checkpoint key hierarchy. `from_safetensors` walks
///  T` at comptime, producing a value of type `T` with every `Tensor`
///  leaf loaded from the `st` file.
///
/// ## Supported field shapes
///
/// - `Tensor`         : required leaf. Looked up at the accumulated field path.
///                       `stz.Error.TensorNotFound` if missing.
/// - `struct { ... }` : recurse into fields, extending the path with
///                      `.field_name`.
/// - `[N]T`           : recurse into elements, extending the path with
///                      `.{index}`.
/// - `?T`             : optional subtree. `stz.Error.TensorNotFound` *anywhere
///                       inside the subtree becomes `null` at the optional
///                       level.* Used for tied weights:
///                       declare `lm_head: ?struct { weight: Tensor }`
///                       and the consumer resolves the alias at use time.
///
/// Any other type is a compile error. No extension points right now, if we
///  need different loading behavior per field, consider extending this walker
///  rather than forking at the call site.
///
/// ## Zero-copy semantics
///
/// When the checkpoint's dtype matches the target dtype, each `Tensor` is
///  constructed as a `.borrow` view into the mmap'd checkpoint bytes.
///
/// When dtypes differ, the walker allocates and converts element-wise
///  (f32/bf16 only). The caller keeps the mmap alive until after the
///  loaded tensors have been uploaded to the device.
pub fn from_safetensors(
    comptime T: type,
    st: *stz.SafeTensorsFile,
    opts: Opts,
) !T {
    return try walk(T, st, opts, "");
}

fn walk(
    comptime T: type,
    st: *stz.SafeTensorsFile,
    opts: Opts,
    comptime prefix: []const u8,
) anyerror!T {
    if (T == Tensor) {
        const view = try st.get(prefix);
        return try tensor_from_view(opts.allocator, opts.dtype, view);
    }
    switch (@typeInfo(T)) {
        .optional => |opt| {
            return walk(opt.child, st, opts, prefix) catch |err| switch (err) {
                stz.Error.TensorNotFound => return null,
                else => return err,
            };
        },
        .@"struct" => |info| {
            var result: T = undefined;
            inline for (info.fields) |field| {
                const sep = if (prefix.len == 0) "" else ".";
                @field(result, field.name) = try walk(
                    field.type,
                    st,
                    opts,
                    prefix ++ sep ++ field.name,
                );
            }
            return result;
        },
        .array => |arr| {
            var result: T = undefined;
            inline for (0..arr.len) |i| {
                const sep = if (prefix.len == 0) "" else ".";
                result[i] = try walk(
                    arr.child,
                    st,
                    opts,
                    prefix ++ sep ++ std.fmt.comptimePrint("{d}", .{i}),
                );
            }
            return result;
        },
        else => @compileError("from_safetensors: unsupported type `" ++ @typeName(T) ++
            "` (expected Tensor, struct, array, or optional of same)"),
    }
}

/// Materialize a `Tensor` from a safetensors `TensorView`.
///
/// Zero-copy borrow when dtypes match, else element-wise allocate-and-convert.
///
/// Cross-dtype conversion currently only `{f32, bf16} <-> {f32, bf16}`.
///  Any other source or target dtype on the mismatch path returns
///  `error.UnsupportedDtypeConversion`.
///
/// The matched-dtype (borrow) path supports the full set in `stz_dtype_matches`.
fn tensor_from_view(
    allocator: std.mem.Allocator,
    target_dtype: DType,
    view: stz.TensorView,
) !Tensor {
    var shape_buf: [8]i64 = undefined;
    if (view.info.shape.len > shape_buf.len) return error.TensorRankUnsupported;
    for (view.info.shape, 0..) |d, i| shape_buf[i] = @intCast(d);
    const shape = shape_buf[0..view.info.shape.len];

    if (stz_dtype_matches(view.info.dtype, target_dtype)) {
        return try Tensor.host(target_dtype, shape, .{ .borrow = view.data });
    }

    if (!convert_supported(view.info.dtype, target_dtype)) {
        return error.UnsupportedDtypeConversion;
    }

    const dst = try Tensor.host(target_dtype, shape, .{ .alloc = allocator });
    const count = dst.shape.num_elements();
    for (0..count) |i| write_element(dst, i, read_element(view, i));
    return dst;
}

/// True iff `tensor_from_view` can convert `src` elements to `dst` dtype
///  element-wise (via the f32 pivot in `read_element`/`write_element`).
inline fn convert_supported(src: stz.Dtype, dst: DType) bool {
    const src_ok = src == .f32 or src == .bf16;
    const dst_ok = dst == .f32 or dst == .bf16;
    return src_ok and dst_ok;
}

inline fn stz_dtype_matches(stz_dt: stz.Dtype, zg_dt: DType) bool {
    return switch (zg_dt) {
        .f32 => stz_dt == .f32,
        .bf16 => stz_dt == .bf16,
        .f16 => stz_dt == .f16,
        .f64 => stz_dt == .f64,
        .i32 => stz_dt == .i32,
        .i64 => stz_dt == .i64,
        inline else => |x| @panic("from_safetensors: unsupported target dtype " ++ @tagName(x)),
    };
}

inline fn read_element(view: stz.TensorView, idx: usize) f32 {
    return switch (view.info.dtype) {
        .f32 => std.mem.bytesAsSlice(f32, view.data)[idx],
        .bf16 => DType.bf16.decode(f32, std.mem.bytesAsSlice(u16, view.data)[idx]),
        else => unreachable,
    };
}

inline fn write_element(buf: Tensor, idx: usize, val: f32) void {
    switch (buf.dtype) {
        .f32 => buf.as_slice(f32)[idx] = val,
        .bf16 => buf.as_slice(u16)[idx] = DType.bf16.encode(f32, val),
        else => unreachable,
    }
}
