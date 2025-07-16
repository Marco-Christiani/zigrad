const std = @import("std");
const zg = @import("zigrad");
const Py = @import("python.zig");
const lib = @import("lib.zig");
const NDTensor = zg.NDTensor;
const Graph = zg.Graph;

pub fn run_tests(T: type, device: zg.DeviceReference) !void {
    try test_add_broadcast(T, device);
    try test_mul_backward(T, device);
    try test_div_backward(T, device);
    try test_matmul_backward(T, device);
    try test_matvec_backward(T, device);
    try test_dot_backward(T, device);
    try test_non_square_matmul(T, device);
    try test_pow(T, device);
    try test_sqrt(T, device);
    try test_segment_sum_csr(T, device);
}

fn test_add_broadcast(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    // Test case 1: (2,3,3) + (3,) -> (2,3,3)
    const t1 = try Tensor.from_slice(device, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 3, 3 }, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 1, 1 }, null, opts);
    defer t2.deinit();

    const t4 = try t1.add(t2);
    defer t4.deinit();

    try t4.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_add_broadcast1");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        \\                   [[0, 1, 2], [3, 4, 5], [6, 7, 8]]],
        \\                  dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        \\t4 = t1 + t2
        \\t4.sum().backward()
        \\t4_result = t4.detach().numpy().flatten()
        \\t2_grad = t2.grad.numpy()
    );

    const expected_result = try py_mod.get_slice(T, "t4_result");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_result, t4.get_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());

    // Test case 2: (2,3,3) + (2,1,3) -> (2,3,3)
    const t5 = try Tensor.from_slice(device, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 2, 1, 3 }, opts);
    defer t5.deinit();

    const t6 = try t1.add(t5);
    defer t6.deinit();

    try t6.backward();

    var py_mod2 = try Py.create_module("test_add_broadcast2");
    defer py_mod2.deinit();

    try py_mod2.run(
        \\import torch
        \\t1 = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        \\                   [[0, 1, 2], [3, 4, 5], [6, 7, 8]]],
        \\                  dtype=torch.float32, requires_grad=True)
        \\t5 = torch.tensor([[[1, 1, 1]], [[1, 1, 1]]], dtype=torch.float32, requires_grad=True)
        \\t6 = t1 + t5
        \\t6.sum().backward()
        \\t6_result = t6.detach().numpy().flatten()
        \\t5_grad = t5.grad.numpy().flatten()
    );

    const expected_result2 = try py_mod2.get_slice(T, "t6_result");
    const expected_t5_grad = try py_mod2.get_slice(T, "t5_grad");

    try std.testing.expectEqualSlices(T, expected_result2, t6.get_data());
    try std.testing.expectEqualSlices(T, expected_t5_grad, t5.assume_grad_data());
}

fn test_mul_backward(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const t1 = try Tensor.from_slice(device, &.{2}, null, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{3}, null, opts);
    defer t2.deinit();

    const t3 = try t1.mul(t2);
    defer t3.deinit();

    try t3.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_mul_backward");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([2.0], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)
        \\t3 = t1 * t2
        \\t3.backward()
        \\t1_grad = t1.grad.numpy()
        \\t2_grad = t2.grad.numpy()
    );

    const expected_t1_grad = try py_mod.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());
}

fn test_div_backward(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const t1 = try Tensor.from_slice(device, &.{ 4, 9 }, null, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 2, 3 }, null, opts);
    defer t2.deinit();

    const t3 = try t1.div(t2);
    defer t3.deinit();

    try t3.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_div_backward");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([4.0, 9.0], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([2.0, 3.0], dtype=torch.float32, requires_grad=True)
        \\t3 = t1 / t2
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy()
        \\t2_grad = t2.grad.numpy()
    );

    const expected_t1_grad = try py_mod.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());
}

fn test_matmul_backward(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 0, 0, 1 }, &.{ 2, 2 }, opts);
    defer t2.deinit();

    // Test case 1: No transpose
    const t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    try t3.backward();

    var py_mod1 = try Py.create_module("test_matmul_no_transpose");
    defer py_mod1.deinit();

    try py_mod1.run(
        \\import torch
        \\t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.matmul(t1, t2)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy().flatten()
    );

    const expected_t1_grad = try py_mod1.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod1.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());

    // Reset gradients
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Test case 2: Transpose A
    const t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();

    try t3_trans_a.backward();

    var py_mod2 = try Py.create_module("test_matmul_transpose_a");
    defer py_mod2.deinit();

    try py_mod2.run(
        \\import torch
        \\t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.matmul(t1.T, t2)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy().flatten()
    );

    const expected_t1_grad_trans_a = try py_mod2.get_slice(T, "t1_grad");
    const expected_t2_grad_trans_a = try py_mod2.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad_trans_a, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad_trans_a, t2.assume_grad_data());

    // Reset gradients
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Test case 3: Transpose B
    const t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();

    try t3_trans_b.backward();

    var py_mod3 = try Py.create_module("test_matmul_transpose_b");
    defer py_mod3.deinit();

    try py_mod3.run(
        \\import torch
        \\t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.matmul(t1, t2.T)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy().flatten()
    );

    const expected_t1_grad_trans_b = try py_mod3.get_slice(T, "t1_grad");
    const expected_t2_grad_trans_b = try py_mod3.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad_trans_b, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad_trans_b, t2.assume_grad_data());

    // Reset gradients
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Test case 4: Transpose both A and B
    const t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();

    try t3_trans_ab.backward();

    var py_mod4 = try Py.create_module("test_matmul_transpose_both");
    defer py_mod4.deinit();

    try py_mod4.run(
        \\import torch
        \\t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.matmul(t1.T, t2.T)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy().flatten()
    );

    const expected_t1_grad_trans_ab = try py_mod4.get_slice(T, "t1_grad");
    const expected_t2_grad_trans_ab = try py_mod4.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad_trans_ab, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad_trans_ab, t2.assume_grad_data());
}

fn test_matvec_backward(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    // [1, 2] [1]   =  [3]
    // [3, 4] [1]      [7]
    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 1 }, &.{2}, opts);
    defer t2.deinit();

    const t3 = try t1.matvec(t2, .{});
    defer t3.deinit();

    try t3.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_matvec_backward");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.mv(t1, t2)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy()
    );

    const expected_t1_grad = try py_mod.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());
}

fn test_dot_backward(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3 }, null, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 4, 5, 6 }, null, opts);
    defer t2.deinit();

    const t3 = try t1.dot(t2);
    defer t3.deinit();

    try t3.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_dot_backward");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.dot(t1, t2)
        \\t3.backward()
        \\t1_grad = t1.grad.numpy()
        \\t2_grad = t2.grad.numpy()
    );

    const expected_t1_grad = try py_mod.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());
}

fn test_non_square_matmul(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    // Test non-square matrix multiplication: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 2, 2, 3 }, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1 }, &.{ 2, 3, 2 }, opts);
    defer t2.deinit();

    const t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    try t3.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_non_square_matmul");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]],
        \\                   [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32, requires_grad=True)
        \\t2 = torch.tensor([[[1, 0], [0, 1], [1, 1]],
        \\                   [[0, 1], [1, 0], [1, 1]]], dtype=torch.float32, requires_grad=True)
        \\t3 = torch.bmm(t1, t2)
        \\t3.sum().backward()
        \\t1_grad = t1.grad.numpy().flatten()
        \\t2_grad = t2.grad.numpy().flatten()
    );

    const expected_t1_grad = try py_mod.get_slice(T, "t1_grad");
    const expected_t2_grad = try py_mod.get_slice(T, "t2_grad");

    try std.testing.expectEqualSlices(T, expected_t1_grad, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_t2_grad, t2.assume_grad_data());
}

fn test_pow(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const x = try Tensor.from_slice(device, &.{ 1.0, 2.0, 3.0, 4.0 }, null, opts);
    defer x.deinit();

    // Forward: y = x^3
    const y = try x.pow(3.0);
    defer y.deinit();

    try y.backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_pow_all_cases");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        \\y = x.pow(3.0)
        \\y.sum().backward()
        \\x_grad = x.grad.numpy()
        \\y_out = y.detach().numpy()
    );

    const expected_out = try py_mod.get_slice(T, "y_out");
    const expected_grad = try py_mod.get_slice(T, "x_grad");

    try std.testing.expectEqualSlices(T, expected_out, y.get_data());
    try std.testing.expectEqualSlices(T, expected_grad, x.assume_grad_data());
}

fn test_sqrt(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const x = try Tensor.from_slice(device, &.{ 1.0, 4.0, 9.0, 16.0 }, null, opts);
    defer x.deinit();

    const y = try x.sqrt();
    defer y.deinit();

    try (try y.sum()).backward();

    // PyTorch verification
    var py_mod = try Py.create_module("test_sqrt");
    defer py_mod.deinit();

    try py_mod.run(
        \\import torch
        \\x = torch.tensor([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        \\y = x.sqrt()
        \\y.sum().backward()
        \\x_grad = x.grad.numpy()
        \\y_out = y.detach().numpy()
    );

    const expected_out = try py_mod.get_slice(T, "y_out");
    const expected_grad = try py_mod.get_slice(T, "x_grad");

    try lib.expectApproxEqRelSlices(T, expected_out, y.get_data(), 1e-5);
    try lib.expectApproxEqRelSlices(T, expected_grad, x.assume_grad_data(), 1e-5);
}

fn test_segment_sum_csr(T: type, device: zg.DeviceReference) !void {
    var graph = Graph.init(std.heap.smp_allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(T);
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    // src = [1.0, 2.0, 3.0, 4.0, 5.0]
    // row_ptr = [0, 2, 5]
    // segments = [[1.0, 2.0], [3.0, 4.0, 5.0]]
    const src = try Tensor.from_slice(device, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 }, null, opts);
    defer src.deinit();

    const row_ptr = [_]usize{ 0, 2, 5 };
    const out = try src.segment_sum_csr(&row_ptr, &.{2});
    defer out.deinit();

    try out.backward();

    // Verify against PyTorch
    var py_mod = try Py.create_module("test_segment_sum_csr");
    defer py_mod.deinit();

    {
        try py_mod.run(
            \\import torch
            \\src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
            \\row_ptr = [0, 2, 5]
            \\y = torch.stack([
            \\    src[row_ptr[0]:row_ptr[1]].sum(),
            \\    src[row_ptr[1]:row_ptr[2]].sum()
            \\])
            \\y.sum().backward()
            \\out = y.detach().numpy()
            \\grad = src.grad.numpy()
        );

        const expected_out = try py_mod.get_slice(T, "out");
        const expected_grad = try py_mod.get_slice(T, "grad");

        try lib.expectApproxEqRelSlices(T, expected_out, out.get_data(), 1e-5);
        try lib.expectApproxEqRelSlices(T, expected_grad, src.assume_grad_data(), 1e-5);
    }

    scope: {
        py_mod.run(
            \\import torch
            \\from torch_scatter import segment_csr
            \\src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
            \\row_ptr = torch.tensor([0, 2, 5], dtype=torch.long)
            \\y = segment_csr(src, row_ptr, reduce='sum')
            \\y.sum().backward()
            \\y_out = y.detach().numpy()
            \\x_grad = src.grad.numpy()
        ) catch |e| {
            std.debug.print(
                \\Got {any} so we likely could not import torch_scatter (python error can be shown with PY_ERRS=1).
                \\We will skip this double-verification test, the previous one still passed.
                \\
            , .{e});
            break :scope;
        };
        const expected_out = try py_mod.get_slice(T, "out");
        const expected_grad = try py_mod.get_slice(T, "grad");

        try lib.expectApproxEqRelSlices(T, expected_out, out.get_data(), 1e-5);
        try lib.expectApproxEqRelSlices(T, expected_grad, src.assume_grad_data(), 1e-5);
    }
}
