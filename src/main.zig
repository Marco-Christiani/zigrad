const std = @import("std");
pub const zg = @import("zigrad");
const mnist = zg.mnist;
const layer = zg.layer;
pub const std_options = zg.std_options;

pub fn transpose(t: anytype) [2]usize {
    const shape = t.get_shape();
    var s: [2]usize = shape[0..2].*;
    std.mem.swap(usize, &s[0], &s[1]);
    return s;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var gpu = zg.device.CudaDevice.init(0, gpa.allocator());
    defer gpu.deinit();

    const m: usize = 3;
    const n: usize = 5;

    const A = try zg.NDTensor(f32).empty(&.{ n, m }, false, gpu.reference());
    defer A.deinit();

    const x = try zg.NDTensor(f32).empty(&.{n}, false, gpu.reference());
    defer x.deinit();

    const y = try zg.NDTensor(f32).empty(&.{3}, false, gpu.reference());
    defer y.deinit();

    gpu.mem_sequence(f32, A.get_data(), 0.0, 1.0);
    gpu.mem_sequence(f32, x.get_data(), 0.0, 1.0);

    gpu.blas.matvec(
        f32,
        A.get_data(),
        x.get_data(),
        y.get_data(),
        n, //as[0],
        m, //as[1],
        true,
        1.0,
        0.0,
    );

    //gpu.blas.matmul(
    //    f32,
    //    A.get_data(),
    //    B.get_data(),
    //    C.get_data(),
    //    m,
    //    n,
    //    n,
    //    transpose,
    //    transpose,
    //    n,
    //    m,
    //    n,
    //    1.0,
    //    0.0,
    //);

    // both transposed
    //gpu.blas.matmul(
    //    f32,
    //    A.get_data(),
    //    B.get_data(),
    //    C.get_data(),
    //    n,
    //    m,
    //    n,
    //    true,
    //    true,
    //    0,
    //    0,
    //    0,
    //    1.0,
    //    0.0,
    //);

    // B transposed
    //gpu.blas.matmul(
    //    f32,
    //    A.get_data(),
    //    B.get_data(),
    //    C.get_data(),
    //    m,
    //    n,
    //    m,
    //    false,
    //    true,
    //    0,
    //    0,
    //    0,
    //    1.0,
    //    0.0,
    //);

    // A transposed
    //gpu.blas.matmul(
    //    f32,
    //    A.get_data(),
    //    B.get_data(),
    //    C.get_data(),
    //    m,
    //    n,
    //    m,
    //    true,
    //    false,
    //    0,
    //    0,
    //    0,
    //    1.0,
    //    0.0,
    //);

    y.print();

    //gpu.sync();

    //gpu.blas.matvec(f32, A.get_data(), x.get_data(), y.get_data(), 5, 5, false, 1.0, 0.0);

    //y.print();
}
