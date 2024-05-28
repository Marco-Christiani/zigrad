const std = @import("std");
const c = @cImport(@cInclude("Accelerate/Accelerate.h"));
// pub fn main() !void {
//     const gemmT = gemm(f32);
//     const n = 4;
//
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     const alloc = arena.allocator();
//     defer arena.deinit();
//     const A = try gemmT.generate(n, &sequential, alloc);
//     const B = try gemmT.generate(n, &rsequential, alloc);
//     const C0 = try gemmT.generate(n, &zeroes, alloc);
//     const C1 = try gemmT.generate(n, &zeroes, alloc);
//     const C2 = try gemmT.generate(n, &zeroes, alloc);
//     var C3 = try gemmT.generate(n, &zeroes, alloc);
//     gemmT.display(n, A);
//     gemmT.display(n, B);
//     gemmT.display(n, C0);
//
//     const Cref = try gemmT.generate(n, &zeroes, alloc);
//     gemmT.agemm(n, A, B, @constCast(Cref));
//     std.debug.print("Ref:\n", .{});
//     gemmT.display(n, Cref);
//
//     var t0 = std.time.nanoTimestamp();
//     gemmT.naive0(n, A, B, @constCast(C0));
//     std.debug.print("\nnaive0 {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
//     gemmT.display(n, C0);
//     std.debug.print("{}\n", .{arrayEqual(f32, Cref, C0)});
//
//     t0 = std.time.nanoTimestamp();
//     gemmT.naive1(n, A, B, @constCast(C1));
//     std.debug.print("naive1 {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
//     gemmT.display(n, C1);
//
//     t0 = std.time.nanoTimestamp();
//     gemmT.inlineManual(n, A, B, @constCast(C2));
//     std.debug.print("inlineManual {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
//     gemmT.display(n, C2);
//
//     inline for (2..5) |q| {
//         const inline_n = 2 << q;
//         t0 = std.time.nanoTimestamp();
//         // gemmT.inlineFor2(n, A, B, @constCast(C3), inline_n);
//         // gemmT.simd(n, A, B, @constCast(C3), inline_n);
//         gemmT.tiled_simd(n, A, B, @constCast(C3), inline_n, 4);
//         std.debug.print("tiled_simd {:<7} {:<10} {:<10} {}\n", .{ n, n * n, std.time.nanoTimestamp() - t0, inline_n });
//         gemmT.display(n, C3);
//         C3 = try gemmT.generate(n, &zeroes, alloc);
//     }
// }

pub fn main() !void {
    const minBits: u16 = 32;
    const maxBits: u16 = 32;
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
    const maxArrSize: f32 = 512;
    const minK: usize = @intFromFloat(@log2(minArrSize) - 1);
    const maxK: usize = @intFromFloat(@log2(maxArrSize));
    var tracker: *BenchmarkTracker = @constCast(&BenchmarkTracker.init());
    std.debug.print("\n{s:<5} {s:<5} {s:<7} {s:<10} {s:<20} {s:<20} {s:<6.4}\n", .{ "i", "type", "n", "N", "fn", "ms", "GFLOPS" });

    @setEvalBranchQuota(1000000);
    inline for (0..types.len) |i| {
        inline for (minK..maxK) |k| {
            const n: u64 = 2 << k;
            std.debug.print("n={}\n", .{n});
            const gemmT = gemm(types[i]);
            const A = try gemmT.generate(n, &sequential, alloc);
            const B = try gemmT.generate(n, &rsequential, alloc);
            const C = try gemmT.generate(n, &zeroes, alloc);
            const Cref = try gemmT.generate(n, &zeroes, alloc);

            tracker.run(types[i], n, "agemm", gemmT.agemm, .{ n, A, B, @constCast(Cref) });

            tracker.run(types[i], n, "naive0", gemmT.naive0, .{ n, A, B, @constCast(C) });
            std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
            zero(types[i], C);

            tracker.run(types[i], n, "naive1", gemmT.naive1, .{ n, A, B, @constCast(C) });
            std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
            zero(types[i], C);

            tracker.run(types[i], n, "inlineManual", gemmT.inlineManual, .{ n, A, B, @constCast(C) });
            std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
            zero(types[i], C);

            tracker.run(types[i], n, "inlineFor", gemmT.inlineFor, .{ n, A, B, @constCast(C) });
            std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
            zero(types[i], C);

            inline for (1..4) |q| {
                const vec_n = 1 << q;
                var nameBuf: [20]u8 = undefined;
                var name = try std.fmt.bufPrint(&nameBuf, "simd-{d}", .{vec_n});
                tracker.run(types[i], n, name, gemmT.simd, .{ n, A, B, @constCast(C), vec_n });
                std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
                // gemmT.display(n, C);
                // gemmT.display(n, Cref);
                zero(types[i], C);

                inline for (q..8) |b| {
                    const block_size = 1 << b;
                    name = try std.fmt.bufPrint(&nameBuf, "tiled_simd-{d}-{d}", .{ vec_n, block_size });
                    tracker.run(types[i], n, name, gemmT.tiled_simd, .{ n, A, B, @constCast(C), vec_n, block_size });
                    std.debug.print("{}\n", .{arrayEqual(types[i], Cref, C)});
                    zero(types[i], C);
                }
            }
        }
    }
    tracker.end();
}

fn calcGflops(n: u64, duration_ns: u64) f64 {
    if (duration_ns == 0) {
        return 0;
    }
    const num: f64 = @floatFromInt(2 * (n * n * n));
    const den: f64 = @floatFromInt(duration_ns);
    return num / den;
}

pub fn zero(comptime T: type, A: []T) void {
    for (A) |*a| {
        a.* = 0;
    }
}

pub fn arrayEqual(comptime T: type, A: []T, B: []T) bool {
    if (A.len != B.len) return false;
    for (A, B) |a, b| {
        if (a != b) return false;
    }
    return true;
}

const BenchmarkTracker = struct {
    const Self = @This();
    timer: *std.time.Timer,
    count: usize,
    best_run: usize,
    peak_gflops: f64,
    print: bool,

    pub fn init() Self {
        const timer = std.time.Timer.start() catch @panic("Failed to start timer");
        return Self{ .timer = @constCast(&timer), .count = 0, .best_run = 0, .peak_gflops = 0, .print = true };
    }

    fn run(
        self: *Self,
        T: type,
        n: u64,
        name: []const u8,
        comptime func: anytype,
        args: anytype,
    ) void {
        self.count += 1;
        self.timer.reset();
        @call(.auto, func, args);
        const duration_ns = self.timer.read();
        const currgflops = calcGflops(n, duration_ns);
        if (self.print) {
            std.debug.print("{:<5} {:<5} {:<7} {:<10} {s:<20} {:<20} {d:<6.4}\n", .{ self.count, T, n, n * n, name, duration_ns / std.time.ns_per_ms, currgflops });
        }
        if (currgflops > self.peak_gflops) {
            self.best_run = self.count;
            self.peak_gflops = currgflops;
        }
    }

    fn end(self: Self) void {
        if (self.print) std.debug.print("Peak Gflops {d}\n", .{self.peak_gflops});
    }
};

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
            _ = vec_n;
            var alloc = std.heap.page_allocator;
            var Bt = alloc.alignedAlloc(T, null, n * n) catch unreachable;
            defer alloc.free(Bt);
            for (0..n) |i| {
                for (0..n) |j| {
                    Bt[j * n + i] = B[i * n + j];
                }
            }
            var in: usize = 0;
            var ij: usize = 0;
            for (0..n) |i| {
                in = i * n;
                for (0..n) |j| {
                    const nj = n * j;
                    ij = in + j;
                    // std.debug.print("a={d} b={d}\n", .{ A[in..(in + n)], Bt[nj..(nj + n)] });
                    C[ij] += @reduce(.Add, @as(@Vector(n, T), A[in..][0..n].*) * @as(@Vector(n, T), Bt[nj..][0..n].*));
                }
            }
        }

        pub fn tiled_simd(comptime n: usize, A: []const T, B: []const T, C: []T, comptime vec_n: usize, comptime block_size: usize) void {
            // if (n_blocks == 0 or n == 0) return;
            // const block_size = (n + n_blocks - 1) / n_blocks;
            // if (block_size == 0) return; // Prevent zero-sized blocks
            // const n_blocks: usize = block_size * (n - 1) / (block_size - 1);
            const n_blocks_per_dim: usize = @intFromFloat(@as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(block_size)));
            const n_blocks = n_blocks_per_dim; // * n_blocks_per_dim;
            // std.debug.print("n={d} block_size={d} n_blocks={d}\n", .{ n, block_size, n_blocks });

            var start_i: usize = 0;
            var end_i: usize = 0;
            var start_j: usize = 0;
            var end_j: usize = 0;
            var start_k: usize = 0;
            var end_k: usize = 0;
            var sum: T = 0;
            var r: usize = 0;

            const VT = @Vector(vec_n, T);

            for (0..n_blocks) |block_i| {
                start_i = block_i * block_size;
                if (start_i >= n) break; // Prevent processing non-existent rows
                end_i = @min(start_i + block_size, n);

                for (0..n_blocks) |block_k| {
                    start_k = block_k * block_size;
                    if (start_k >= n) break; // Prevent processing non-existent depths
                    end_k = @min(start_k + block_size, n);
                    for (0..n_blocks) |block_j| {
                        start_j = block_j * block_size;
                        if (start_j >= n) break; // Prevent processing non-existent columns
                        end_j = @min(start_j + block_size, n);

                        // pack(end_k - start_k, A[start_i * n + start_k ..], n, packedA[0..]);
                        // pack(end_k - start_k, B[start_k * n + start_j ..], n, packedB[0..]);
                        // for (start_i..end_i) |i| {
                        //     for (start_j..end_j) |j| {
                        //         sum = C[i * n + j];
                        //         r = 0;
                        //
                        //         while (r + vec_n <= end_k - start_k) {
                        //             sum += @reduce(.Add, @as(@Vector(vec_n, f64), packedA[r * block_size ..][0..vec_n].*) * @as(@Vector(vec_n, f64), packedB[r * block_size ..][0..vec_n].*));
                        //             r += vec_n;
                        //         }
                        //
                        //         while (r < end_k - start_k) {
                        //             sum += packedA[r] * packedB[r];
                        //             r += 1;
                        //         }
                        //
                        //         C[i * n + j] = sum;
                        //     }
                        // }
                        for (start_j..end_j) |j| {
                            for (start_i..end_i) |i| {
                                sum = C[i * n + j];
                                r = start_k;
                                const rem = (end_k - start_k) % vec_n;

                                // while (r + vec_n <= end_k) {
                                while (r < (end_k - rem)) {
                                    sum += @reduce(.Add, @as(VT, A[(i * n + r)..][0..vec_n].*) * @as(VT, B[(r * n + j)..][0..vec_n].*));
                                    // sum += @reduce(.Add, @as(VT, packedA[r..][0..vec_n].*) * @as(VT, packedB[r..][0..vec_n].*));
                                    r += vec_n;
                                }

                                // while (r < end_k) {
                                for (0..rem) |_| {
                                    sum += A[i * n + r] * B[r * n + j];
                                    // sum += packedA[r] * packedB[r];
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

        pub fn agemm(n: usize, A: []const T, B: []const T, C: []T) void {
            c.cblas_sgemm(
                c.CblasRowMajor,
                c.CblasNoTrans,
                c.CblasNoTrans,
                @intCast(n),
                @intCast(n),
                @intCast(n),
                1.0,
                A.ptr,
                @intCast(n),
                B.ptr,
                @intCast(n),
                1.0,
                C.ptr,
                @intCast(n),
            );
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
    const gemmT = gemm(f32);
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

    // const Cref = try gemmT.generate(n, &zeroes, alloc);
    // gemmT.agemm(n, A, B, @constCast(Cref));
    // std.debug.print("Ref:\n", .{});
    // gemmT.display(n, Cref);

    var t0 = std.time.nanoTimestamp();
    gemmT.naive0(n, A, B, @constCast(C0));
    std.debug.print("\nnaive0 {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C0);

    t0 = std.time.nanoTimestamp();
    gemmT.naive1(n, A, B, @constCast(C1));
    std.debug.print("naive1 {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C1);

    t0 = std.time.nanoTimestamp();
    gemmT.inlineManual(n, A, B, @constCast(C2));
    std.debug.print("inlineManual {:<7} {:<10} {:<10}\n", .{ n, n * n, std.time.nanoTimestamp() - t0 });
    gemmT.display(n, C2);

    inline for (2..5) |q| {
        const inline_n = 2 << q;
        t0 = std.time.nanoTimestamp();
        // gemmT.inlineFor2(n, A, B, @constCast(C3), inline_n);
        // gemmT.simd(n, A, B, @constCast(C3), inline_n);
        gemmT.tiled_simd(n, A, B, @constCast(C3), inline_n, 4);
        std.debug.print("tiled_simd {:<7} {:<10} {:<10} {}\n", .{ n, n * n, std.time.nanoTimestamp() - t0, inline_n });
        gemmT.display(n, C3);
        C3 = try gemmT.generate(n, &zeroes, alloc);
    }
}
