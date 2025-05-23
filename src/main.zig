const std = @import("std");

const zg = @import("zigrad");

const Tensor = zg.NDTensor(f32);
const Array = zg.NDArray(f32);

// It is expected to have some epsilon between the
// device and the host. Floating point is not associative
// and the GPU does perform oprations in the same order.
const epsilon: f32 = 1e-3;
pub fn similar(x: *Tensor, y: *Tensor) !void {
    return similar_slice(x.get_data(), y.get_data());
}
pub fn similar_slice(x: []const f32, y: []const f32) error{ NotSimilar, WrongSize }!void {
    if (x.len != y.len) return error.WrongSize;
    for (x, y) |u, v| if (@abs(u - v) > epsilon) return error.NotSimilar;
}

pub fn main() !void {
    if (comptime zg.backend == .HOST) {
        std.log.info("Nothing to do, yet.", .{});
    } else {
        try test_host_cuda_consistency();
    }
}

fn test_host_cuda_consistency() !void {
    if (comptime zg.backend != .CUDA) {
        @compileError("Zigrad backend must be targeted at CUDA to run cuda_tests.zig");
    }
    //var prng = std.Random.DefaultPrng.init(zg.settings.seed);
    //const rand = prng.random();
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    var gpu = zg.device.CudaDevice.init(0);
    defer gpu.deinit();

    var gm = zg.GraphManager.init(std.heap.smp_allocator, .{});
    defer gm.deinit();

    { // MEMORY TRANSFER //
        std.log.info("TESTING: MEMORY TRANSFER", .{});
        const x = try Tensor.random(&.{256}, .uniform, .{ .device = cpu.reference(), .node_allocator = gm.heap() });
        defer x.deinit();

        const y = try x.to_device(gpu.reference());
        defer y.deinit();

        const z = try y.to_device(cpu.reference());
        defer z.deinit();

        gpu.sync();

        try similar(x, z);
    }

    { // ELEMENTWISE OPS //
        const a = try Tensor.random(&.{256}, .uniform, .{ .device = cpu.reference(), .node_allocator = gm.heap() });
        defer a.deinit();
        const b = try Tensor.random(&.{256}, .uniform, .{ .device = cpu.reference(), .node_allocator = gm.heap() });
        defer b.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();
        const y = try b.to_device(gpu.reference());
        defer y.deinit();

        { // ADDITION //
            std.log.info("TESTING: ELEMENTWISE ADD", .{});
            const c = try a.add(b);
            defer c.deinit();

            const z = try x.add(y);
            defer z.deinit();

            const d = try z.to_device(cpu.reference());
            defer d.deinit();

            gpu.sync();

            try similar(c, d);
        }

        { // SUBTRACTION //
            std.log.info("TESTING: ELEMENTWISE SUB", .{});
            const c = try a.sub(b);
            defer c.deinit();

            const z = try x.sub(y);
            defer z.deinit();

            const d = try z.to_device(cpu.reference());
            defer d.deinit();

            gpu.sync();

            try similar(c, d);
        }

        { // MULTIPLICATION //
            std.log.info("TESTING: ELEMENTWISE MUL", .{});
            const c = try a.mul(b);
            defer c.deinit();

            const z = try x.mul(y);
            defer z.deinit();

            const d = try z.to_device(cpu.reference());
            defer d.deinit();

            gpu.sync();

            try similar(c, d);
        }
    }

    cpu.clear_cache();
    gpu.clear_cache();
    gm.reset();

    { // BATCH MATRIX MUL //
        std.log.info("TESTING: BATCH MATRIX MUL", .{});
        const a = try Tensor.random(
            &.{ 3, 256, 256 },
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer a.deinit();
        const b = try Tensor.random(
            &.{ 3, 256, 256 },
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer b.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();
        const y = try b.to_device(gpu.reference());
        defer y.deinit();

        for (
            &[_]bool{ true, true, false, false },
            &[_]bool{ true, false, true, false },
        ) |trans_a, trans_b| {
            const c = try a.bmm(b, .{
                .trans_a = trans_a,
                .trans_b = trans_b,
            });
            defer c.deinit();

            const z = try x.bmm(y, .{
                .trans_a = trans_a,
                .trans_b = trans_b,
            });
            defer z.deinit();

            const d = try z.to_device(cpu.reference());
            defer d.deinit();

            try similar(c, d);
        }
    }

    cpu.clear_cache();
    gpu.clear_cache();
    gm.reset();

    { // MATRIX VECTOR OP //
        std.log.info("TESTING: MATRIX VECTOR MUL", .{});
        const a = try Tensor.random(
            &.{ 256, 256 },
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer a.deinit();
        const b = try Tensor.random(
            &.{256},
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer b.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();
        const y = try b.to_device(gpu.reference());
        defer y.deinit();

        const c = try a.matvec(b, .{});
        defer c.deinit();

        const z = try x.matvec(y, .{});
        defer z.deinit();

        const d = try z.to_device(cpu.reference());
        defer d.deinit();

        try similar(c, d);
    }

    { // SUM ALONG - TODO: Move this to NDArray testing file //
        std.log.info("TESTING: MAX", .{});
        const a = try Tensor.random(
            &.{256},
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer a.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();

        var c = try a.max();
        defer c.deinit();

        var z = try x.max();
        defer z.deinit();

        const d = try z.to_device(cpu.reference());
        defer d.deinit();

        try similar(c, d);
    }

    { // SUM ALONG - TODO: Move this to NDArray testing file //
        std.log.info("TESTING: SUM ALONG", .{});
        const a = try Tensor.random(
            &.{ 256, 256 },
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer a.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();

        var c = try a.data.sum_along(cpu.reference(), .{ .dim = 0 });
        defer c.deinit(cpu.reference());

        var z = try x.data.sum_along(gpu.reference(), .{ .dim = 0 });
        defer z.deinit(gpu.reference());

        const dst = try cpu.mem_alloc(f32, 256);
        defer cpu.mem_free(dst);

        gpu.mem_transfer(f32, z.data, dst, .DtoH);

        try similar_slice(c.data, dst);
    }

    { // SUM ALONG - TODO: Move this to NDArray testing file //
        std.log.info("TESTING: MAX ALONG", .{});
        const a = try Tensor.random(
            &.{ 256, 256 },
            .normal,
            .{ .device = cpu.reference(), .node_allocator = gm.heap() },
        );
        defer a.deinit();

        const x = try a.to_device(gpu.reference());
        defer x.deinit();

        var c = try a.data.max_along(cpu.reference(), .{ .dim = 0 });
        defer c.deinit(cpu.reference());

        var z = try x.data.max_along(gpu.reference(), .{ .dim = 0 });
        defer z.deinit(gpu.reference());

        const dst = try cpu.mem_alloc(f32, 256);
        defer cpu.mem_free(dst);

        gpu.mem_transfer(f32, z.data, dst, .DtoH);

        try similar_slice(c.data, dst);
    }
}
