const std = @import("std");

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
