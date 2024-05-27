const std = @import("std");

var peakgflops: f64 = 0;

pub fn main() !void {
    const minBits: u16 = 16;
    const maxBits: u16 = 64;
    const len = comptime blk: {
        const len: u16 = @as(u16, @intFromFloat(@log2(@as(f16, @floatFromInt(maxBits))))) - @as(u16, @intFromFloat(@log2(@as(f16, @floatFromInt(minBits))))) + 1;
        break :blk len;
    };
    const types: [len]type = comptime blk: {
        var types: [len]type = undefined;
        var i: u16 = 0;
        var bits = minBits;
        while (bits <= maxBits) : (bits *= 2) {
            types[i] = @Type(.{ .Float = .{ .bits = bits } });
            i += 1;
        }
        break :blk types;
    };

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const minArrSize: f32 = 512;
    const maxArrSize: f32 = 4096;
    const minK: usize = @intFromFloat(@log2(minArrSize) - 1);
    const maxK: usize = @intFromFloat(@log2(maxArrSize));
    std.debug.print("\n{s:<5} {s:<7} {s:<10} {s:<20} {s:<20} {s:<6.4}\n", .{ "type", "n", "N", "fn", "ms", "GFLOPS" });
    @setEvalBranchQuota(1000000);
    inline for (0..types.len) |i| {
        inline for (minK..maxK) |k| {
            const n: u64 = 2 << k;
            const gemmT = gemm(types[i]);
            const A = try gemmT.generate(n, &sequential, alloc);
            const B = try gemmT.generate(n, &rsequential, alloc);
            const C = try gemmT.generate(n, &zeroes, alloc);

            const t0 = std.time.nanoTimestamp();
            gemmT.naive0(n, A, B, @constCast(C));
            const t1: i64 = @intCast(std.time.nanoTimestamp());
            const duration0 = @as(u64, @intCast(t1 - t0));

            gemmT.naive1(n, A, B, @constCast(C));
            const t2 = std.time.nanoTimestamp();
            const duration1 = @as(u64, @intCast(t2 - t1));

            gemmT.inlineManual(n, A, B, @constCast(C));
            const t3 = std.time.nanoTimestamp();
            const duration2 = @as(u64, @intCast(t3 - t2));

            gemmT.inlineFor(n, A, B, @constCast(C));
            const t4 = std.time.nanoTimestamp();
            const duration3 = @as(u64, @intCast(t4 - t3));

            format(types[i], n, "naive0", duration0);
            format(types[i], n, "naive1", duration1);
            format(types[i], n, "inlineManual", duration2);
            format(types[i], n, "inlineFor", duration3);

            inline for (1..4) |q| {
                const vec_n = 1 << q;

                var simd_t0 = std.time.nanoTimestamp();
                gemmT.simd(n, A, B, @constCast(C), vec_n);
                var simd_duration = @as(u64, @intCast(std.time.nanoTimestamp() - simd_t0));
                var nameBuf: [20]u8 = undefined;
                var name = try std.fmt.bufPrint(&nameBuf, "tiled-{d}", .{vec_n});
                // std.debug.print("{:<5} {:<7} {:<10} {s}-{:<10} {:<10} {:<6}\n", .{ types[i], n, n * n, "simd", inline_n, simd_duration, calcGflops(n, simd_duration) });
                format(types[i], n, "simd", simd_duration);
                inline for (q..8) |b| {
                    const block_size = 1 << b;
                    simd_t0 = std.time.nanoTimestamp();
                    gemmT.tiled_simd(n, A, B, @constCast(C), vec_n, block_size);
                    simd_duration = @as(u64, @intCast(std.time.nanoTimestamp() - simd_t0));
                    name = try std.fmt.bufPrint(&nameBuf, "tiled_simd-{d}-{d}", .{ vec_n, block_size });
                    format(types[i], n, name, simd_duration);
                }
            }
        }
    }

    std.debug.print("Peak Gflops {d}\n", .{peakgflops});
}

fn format(
    T: type,
    n: u64,
    name: []const u8,
    duration_ns: u64,
) void {
    const currgflops = calcGflops(n, duration_ns);
    std.debug.print("{:<5} {:<7} {:<10} {s:<20} {:<20} {d:<6.4}\n", .{ T, n, n * n, name, duration_ns / std.time.ns_per_ms, currgflops });
    if (currgflops > peakgflops) peakgflops = currgflops;
}

fn calcGflops(n: u64, duration_ns: u64) f64 {
    if (duration_ns == 0) {
        return 0;
    }
    const num: f64 = @floatFromInt(2 * (n * n * n));
    const den: f64 = @floatFromInt(duration_ns);
    return num / den;
}

/// Multiply n x n matrices
pub fn gemm(comptime T: type) type {
    return struct {
        pub fn naive0(comptime n: usize, A: []const T, B: []const T, C: []T) void {
            // n is effectively the stride
            for (0..n) |i| {
                for (0..n) |j| {
                    for (0..n) |r| {
                        const ij = i * n + j;
                        const ir = i * n + r;
                        const rj = r * n + j;
                        C[ij] += A[ir] * B[rj];
                        // std.debug.print("({},{})={}, ({},{})={} ({},{})={}\n", .{ i, j, ij, i, r, ir, r, j, rj });
                    }
                }
            }
        }

        pub fn naive1(comptime n: usize, A: []const T, B: []const T, C: []T) void {
            var in: usize = 0;
            var ij: usize = 0;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    ij = in + j;
                    for (0..n) |r| {
                        C[ij] += A[in + r] * B[r * n + j];
                    }
                }
            }
        }

        pub fn inlineManual(comptime n: usize, A: []const T, B: []const T, C: []T) void {
            var in: usize = 0;
            var ij: usize = 0;
            var r: usize = 0;
            const rem = n % 4;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    ij = in + j;
                    r = 0;
                    while (r < (n - rem)) : (r += 4) {
                        C[ij] += A[in + r] * B[r * n + j];
                        C[ij] += A[in + r + 1] * B[(r + 1) * n + j];
                        C[ij] += A[in + r + 2] * B[(r + 2) * n + j];
                        C[ij] += A[in + r + 3] * B[(r + 3) * n + j];
                    }

                    for (r..n) |q| {
                        C[ij] += A[in + q] * B[q * n + j];
                    }
                }
            }
        }

        pub fn inlineFor(comptime n: usize, A: []const T, B: []const T, C: []T) void {
            var in: usize = 0;
            var ij: usize = 0;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    ij = in + j;
                    inline for (0..n) |r| {
                        C[ij] += A[in + r] * B[r * n + j];
                    }
                }
            }
        }

        pub fn inlineFor2(comptime n: usize, A: []const T, B: []const T, C: []T, comptime inline_n: usize) void {
            var in: usize = 0;
            var ij: usize = 0;
            var r: usize = 0;
            const rem = n % inline_n;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    ij = in + j;
                    r = 0;
                    while (r < (n - rem)) {
                        inline for (0..inline_n) |_| {
                            C[ij] += A[in + r] * B[r * n + j];
                            r += 1;
                        }
                    }

                    inline for (0..rem) |_| {
                        C[ij] += A[in + r] * B[r * n + j];
                        r += 1;
                    }
                }
            }
        }

        pub fn simd(comptime n: usize, A: []const T, B: []const T, C: []T, comptime vec_n: usize) void {
            var in: usize = 0;
            var ij: usize = 0;
            var r: usize = 0;
            const rem = n % vec_n;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    ij = in + j;
                    r = 0;
                    while (r < (n - rem)) {
                        C[ij] += @reduce(.Add, @as(@Vector(vec_n, T), A[(in + r)..][0..vec_n].*) * @as(@Vector(vec_n, T), B[(r * n + j)..][0..vec_n].*));
                        r += vec_n;
                    }

                    inline for (0..rem) |_| {
                        C[ij] += A[in + r] * B[r * n + j];
                        r += 1;
                    }
                }
            }
        }

        pub fn tiled_simd(comptime n: usize, A: []const T, B: []const T, C: []T, comptime vec_n: usize, comptime block_size: usize) void {
            // if (n_blocks == 0 or n == 0) return;
            // const block_size = (n + n_blocks - 1) / n_blocks;
            // if (block_size == 0) return; // Prevent zero-sized blocks
            // const n_blocks: usize = block_size * (n - 1) / (block_size - 1);
            const n_blocks_per_dim: usize = @intFromFloat(@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(block_size)));
            const n_blocks = n_blocks_per_dim * n_blocks_per_dim;
            // std.debug.print("n={d} block_size={d} n_blocks={d}\n", .{ n, block_size, n_blocks });

            var start_i: usize = 0;
            var end_i: usize = 0;
            var start_j: usize = 0;
            var end_j: usize = 0;
            var start_k: usize = 0;
            var end_k: usize = 0;
            var sum: T = 0;
            var r: usize = 0;
            for (0..n_blocks) |block_i| {
                start_i = block_i * block_size;
                if (start_i >= n) break; // Prevent processing non-existent rows
                end_i = @min(start_i + block_size, n);

                for (0..n_blocks) |block_j| {
                    start_j = block_j * block_size;
                    if (start_j >= n) break; // Prevent processing non-existent columns
                    end_j = @min(start_j + block_size, n);

                    for (0..n_blocks) |block_k| {
                        start_k = block_k * block_size;
                        if (start_k >= n) break; // Prevent processing non-existent depths
                        end_k = @min(start_k + block_size, n);

                        for (start_i..end_i) |i| {
                            for (start_j..end_j) |j| {
                                sum = C[i * n + j];
                                r = start_k;

                                while (r + vec_n <= end_k) {
                                    sum += @reduce(.Add, @as(@Vector(vec_n, T), A[(i * n + r)..][0..vec_n].*) * @as(@Vector(vec_n, T), B[(r * n + j)..][0..vec_n].*));
                                    r += vec_n;
                                }

                                while (r < end_k) {
                                    sum += A[i * n + r] * B[r * n + j];
                                    r += 1;
                                }

                                C[i * n + j] = sum;
                            }
                        }
                    }
                }
            }
        }

        pub fn tiled(n: usize, A: []const T, B: []const T, C: []T, n_blocks: usize) void {
            // Safety checks
            if (n_blocks == 0 or n == 0) return;

            const block_size = (n + n_blocks - 1) / n_blocks;

            if (block_size == 0) return; // Prevent zero-sized blocks

            for (0..n_blocks) |block_i| {
                const start_i: usize = block_i * block_size;
                if (start_i >= n) break; // Prevent processing non-existent rows
                const end_i: usize = @min(start_i + block_size, n);

                for (0..n_blocks) |block_j| {
                    const start_j: usize = block_j * block_size;
                    if (start_j >= n) break; // Prevent processing non-existent columns
                    const end_j: usize = @min(start_j + block_size, n);

                    for (0..n_blocks) |block_k| {
                        const start_k: usize = block_k * block_size;
                        if (start_k >= n) break; // Prevent processing non-existent depths
                        const end_k: usize = @min(start_k + block_size, n);

                        for (start_i..end_i) |i| {
                            for (start_j..end_j) |j| {
                                for (start_k..end_k) |k| {
                                    const ij = i * n + j;
                                    const ik = i * n + k;
                                    const kj = k * n + j;
                                    C[ij] += A[ik] * B[kj];
                                }
                            }
                        }
                    }
                }
            }
        }

        pub fn generate(comptime n: usize, func: *const fn (type, usize, usize) T, allocator: std.mem.Allocator) ![]T {
            var result = try allocator.alignedAlloc(T, null, n * n);
            // var result = try allocator.alloc(T, n * n);
            for (0..n * n) |i| {
                result[i] = func(T, i, n * n);
            }
            return result;
        }

        pub fn display(n: usize, A: []const T) void {
            std.debug.print("\n", .{});
            for (0..n) |i| {
                std.debug.print("[", .{});
                for (0..n) |j| {
                    const pos = i * n + j;
                    // std.debug.print("({},{}) ({}): {}\n", .{ i, j, pos, A[pos] });
                    std.debug.print("{d: >8.2}", .{A[pos]});
                    if ((pos + 1) % n == 0) {
                        std.debug.print("]\n", .{});
                    } else {
                        std.debug.print(", ", .{});
                    }
                }
            }
        }
    };
}

fn sequential(T: type, i: usize, size: usize) T {
    _ = size;
    return @floatFromInt(i);
}

fn rsequential(T: type, i: usize, size: usize) T {
    return @floatFromInt(size - i - 1);
}

fn zeroes(T: type, i: usize, size: usize) T {
    _ = i;
    _ = size;
    return 0.0;
}

test "gemm/display" {
    const gemmT = gemm(f64);
    const n = 4;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const A = try gemmT.generate(n, &sequential, alloc);
    const B = try gemmT.generate(n, &rsequential, alloc);
    const C0 = try gemmT.generate(n, &zeroes, alloc);
    const C1 = try gemmT.generate(n, &zeroes, alloc);
    const C2 = try gemmT.generate(n, &zeroes, alloc);
    var C3 = try gemmT.generate(n, &zeroes, alloc);
    gemmT.display(n, A);
    gemmT.display(n, B);
    gemmT.display(n, C0);

    var t0 = std.time.nanoTimestamp();
    gemmT.naive0(n, A, B, @constCast(C0));
    std.debug.print("\n{:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C0);

    t0 = std.time.nanoTimestamp();
    gemmT.naive1(n, A, B, @constCast(C1));
    std.debug.print("{:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C1);

    t0 = std.time.nanoTimestamp();
    gemmT.inlineManual(n, A, B, @constCast(C2));
    std.debug.print("{:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C2);

    inline for (2..5) |q| {
        const inline_n = 2 << q;
        t0 = std.time.nanoTimestamp();
        // gemmT.inlineFor2(n, A, B, @constCast(C3), inline_n);
        // gemmT.simd(n, A, B, @constCast(C3), inline_n);
        gemmT.tiled_simd(n, A, B, @constCast(C3), inline_n, 4);
        std.debug.print("{:<7} {:<10} {:<10} {}\n", .{ n, n * n, std.time.nanoTimestamp() - t0, inline_n });
        gemmT.display(n, C3);
        C3 = try gemmT.generate(n, &zeroes, alloc);
    }
}
