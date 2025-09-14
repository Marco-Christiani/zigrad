const std = @import("std");
const HostDevice = @import("../host_device.zig");
const opspec = @import("../opspec.zig");

pub fn unbroadcast(self: *HostDevice, T: type, p: opspec.unbroadcast(T)) void {
    const Array = std.BoundedArray(usize, 8);

    if (p.x.len == p.y.len) {
        return scaled_copy(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }

    // remove any leading ones because they contribute nothing to the reduction
    var x_shape = blk: {
        const trimmed = std.mem.trimLeft(usize, p.x_shape, &.{1});
        break :blk Array.fromSlice(trimmed) catch unreachable;
    };

    var y_shape = blk: {
        const trimmed = std.mem.trim(usize, p.y_shape, &.{1});
        break :blk Array.fromSlice(trimmed) catch unreachable;
    };

    // check if we have to deal with any body reductions
    var ones = std.mem.count(usize, y_shape.slice(), &.{1});

    if (x_shape.len > y_shape.len) {
        const dif = x_shape.len - y_shape.len;

        fold_rows(T, .{
            .x = p.x,
            .y = if (ones == 0) p.y else p.scratch,
            .row = _prod(x_shape.slice()[0..dif]),
            .col = _prod(x_shape.slice()[dif..]),
            .alpha = p.alpha,
            .beta = p.beta,
        });

        if (ones == 0) return;

        // remove the indices we just reduced for next round
        x_shape = Array.fromSlice(x_shape.slice()[dif..]) catch unreachable;
    }

    var i: usize = 0;
    while (ones > 0 and i < x_shape.len) {
        if (y_shape.get(i) == 1 and x_shape.get(i) != 1) {
            // TODO: optimize this by detecting streams of 1's in the y_shape
            // and reduce all of those together in one move.

            const x_data = if (x_shape.len == p.x_shape.len) p.x else p.scratch;
            const y_data = if (ones == 1) p.y else p.scratch;

            self.sum_along(T, .{
                .x = x_data,
                .x_shape = x_shape.slice(),
                .y = y_data,
                .y_shape = y_shape.slice(),
                .dim = i,
                .alpha = p.alpha,
                .beta = p.beta,
            });

            _ = x_shape.orderedRemove(i);
            _ = y_shape.orderedRemove(i);
            ones -= 1;
            continue;
        }
        i += 1; // only increment if didn't remove an index
    }
}

pub fn broadcast(_: *const HostDevice, T: type, p: opspec.broadcast(T)) void {
    if (p.x.len == p.y.len) {
        scaled_copy(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }

    // TODO: make a better broadcast
    for (0..p.y.len) |i| p.y[i] = p.alpha * p.x[i % p.x.len] + p.beta * p.y[i % p.y.len];
}

// TODO: Replace this with general reduce
pub fn sum_along(_: *const HostDevice, T: type, p: opspec.sum_along(T)) void {
    std.debug.assert(0 < p.x_shape.len);
    std.debug.assert(p.dim < p.x_shape.len);

    // flat, head, and tail reduce base cases
    if (p.y.len == 1) {
        return flat_reduce(T, .{
            .x = p.x,
            .y = p.y,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    } else if (p.dim == 0) {
        return fold_rows(T, .{
            .x = p.x,
            .y = p.y,
            .row = p.x_shape[0],
            .col = _prod(p.x_shape[1..]),
            .alpha = p.alpha,
            .beta = p.beta,
        });
    } else if (p.dim + 1 == p.x_shape.len) {
        return fold_cols(T, .{
            .x = p.x,
            .y = p.y,
            .row = _prod(p.x_shape[0..p.dim]),
            .col = p.x_shape[p.dim],
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }
    // body reduction - we can always imagine that we have an ijk
    // tensor where j is the value we want to reduce. This works
    // because we require flat and symmetric memory layout.
    const n_chunks = _prod(p.x_shape[0..p.dim]);
    const y_chunk_size = _prod(p.x_shape[p.dim + 1 ..]);
    const x_chunk_size = p.x_shape[p.dim] * y_chunk_size;

    for (0..n_chunks) |n| {
        fold_rows(T, .{
            .x = p.x[x_chunk_size * n ..][0..x_chunk_size],
            .y = p.y[y_chunk_size * n ..][0..y_chunk_size],
            .row = p.x_shape[p.dim],
            .col = y_chunk_size,
            .alpha = p.alpha,
            .beta = p.beta,
        });
    }
}

// TODO: Should this ever be mixed with reduce? Seems like a bad idea
// because certain optimizations actually have extra data that general
// reduce (using addition) doesn't have.
pub fn max_along(_: *const HostDevice, T: type, p: opspec.max_along(T)) void {
    std.debug.assert(p.dim < p.x_shape.len);

    const max_dim_size = p.x_shape[p.dim];

    var slice_size: usize = 1;
    for (p.dim + 1..p.x_shape.len) |i| {
        slice_size *= p.x_shape[i];
    }

    for (0..p.y.len) |i| {
        var max_val: T = -std.math.inf(T);
        const base_offs = (i / slice_size) * (slice_size * max_dim_size) + (i % slice_size);
        for (0..max_dim_size) |j| { // can be optimized if the view along this dim is contiguous (just check dim stride)
            const curr_offs = base_offs + j * slice_size;
            const curr_val = p.x[curr_offs];
            if (curr_val > max_val) {
                max_val = curr_val;
            }
        }
        p.y[i] = max_val;
    }
}

pub fn scaled_copy(T: type, p: struct {
    x: []const T,
    y: []T,
    alpha: T,
    beta: T,
}) void {
    std.debug.assert(p.x.len == p.y.len);
    for (p.x, p.y) |x, *y| y.* = p.alpha * x + p.beta * y.*;
}

pub fn flat_reduce(T: type, p: struct {
    x: []const T,
    y: []T,
    alpha: T,
    beta: T,
}) void {
    std.debug.assert(p.x.len > 0);
    std.debug.assert(p.y.len == 1);

    var s: T = 0.0;
    var i: usize = 0;

    if (comptime std.simd.suggestVectorLength(T)) |N| {

        // create a vector of all zeros
        var u: @Vector(N, T) = @splat(0);

        // check if we can fit another vector in.
        // if we can, add it to our running reductions
        while ((i + N) <= p.x.len) : (i += N) {
            const v: @Vector(N, T) = p.x[i..][0..N].*;
            u += v;
        }

        // reduce the vector to a single element
        s = @reduce(.Add, u);
    }

    while (i < p.x.len) : (i += 1) {
        s += p.x[i];
    }

    p.y[0] = p.alpha * s + p.beta * p.y[0];
}

/// (M,N) -> (1,N)
pub fn fold_rows(T: type, p: struct {
    x: []const T,
    y: []T,
    row: usize,
    col: usize,
    alpha: T = 1.0,
    beta: T = 0.0,
}) void {
    std.debug.assert(p.x.len == p.col * p.row);
    std.debug.assert(p.y.len >= p.col);

    const N = comptime std.simd.suggestVectorLength(T) orelse unreachable;
    const M: usize = 32;
    const row = p.row;
    const col = p.col;
    var offset: usize = 0;

    var vec_arr: [M]@Vector(N, T) = undefined;
    var scl_arr: [N]T = undefined;

    const _a: @Vector(N, T) = @splat(p.alpha);
    const _b: @Vector(N, T) = @splat(p.beta);
    const rem = col % N;

    // start by priming the remainder
    for (col - rem..col, 0..) |i, j| {
        scl_arr[j] = p.x[i];
    }

    while (offset < col) {
        { // load array first to reuse values
            var j: usize = offset;
            var k: usize = 0;
            while (k < M and (j + N) <= col) : ({
                j += N;
                k += 1;
            }) {
                vec_arr[k] = p.x[j..][0..N].*;
            }
        }

        var j: usize = 0;
        var k: usize = 0;
        for (1..row) |i| {
            j = offset;
            k = 0;
            const x_row = p.x[i * col ..][0..col];
            while (k < M and (j + N) <= col) : ({
                j += N;
                k += 1;
            }) {
                const u: @Vector(N, T) = x_row[j..][0..N].*;
                vec_arr[k] += u;
            }

            // if no more SIMD vectors fit within the
            // column, we finish out the remainder
            if (col <= j + N) {
                var s: usize = 0;
                while (j < col) : ({
                    j += 1;
                    s += 1;
                }) {
                    scl_arr[s] += x_row[j];
                }
            }
        }

        for (0..k) |i| {
            const y: @Vector(N, T) = p.y[offset + i * N ..][0..N].*;
            p.y[offset + i * N ..][0..N].* = _a * vec_arr[i] + _b * y;
        }

        offset = j;
    }

    for (col - rem..col, 0..) |i, j| {
        p.y[i] = p.alpha * scl_arr[j] + p.beta * p.y[i];
    }
}

/// (M,N) -> (M,1)
pub fn fold_cols(T: type, p: struct {
    x: []const T,
    y: []T,
    row: usize,
    col: usize,
    alpha: T = 1.0,
    beta: T = 1.0,
}) void {
    std.debug.assert(p.x.len == p.col * p.row);
    std.debug.assert(p.y.len >= p.row);
    const row = p.row;
    const col = p.col;

    for (0..row) |i| {
        const x_row = p.x[i * col ..][0..col];

        var s: T = 0;
        var j: usize = 0;
        if (comptime std.simd.suggestVectorLength(T)) |N| {
            var u: @Vector(N, T) = @splat(0);
            while ((j + N) <= col) : (j += N) {
                const v: @Vector(N, T) = x_row[j..][0..N].*;
                u += v;
            }
            s += @reduce(.Add, u);
        }

        while (j < col) : (j += 1) {
            s += x_row[j];
        }

        p.y[i] = p.alpha * s + p.beta * p.y[i];
    }
}

fn _prod(sizes: []const usize) usize {
    var n: usize = 1;
    for (sizes) |m| n *= m;
    return n;
}
