const std = @import("std");

const ndtensor = @import("../ndtensor.zig");
const NDTensor = ndtensor.NDTensor;
const Op = ndtensor.Op;

/// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
pub fn max(T: type, self: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    const max_val = try self.data.max(allocator);
    return try NDTensor(T).createDependent(.{
        .data = max_val,
        .op = .MAX,
        .children = &[_]*const NDTensor(T){self},
        .requires_grad = self.requires_grad,
        ._backward = _max_backward,
        .allocator = allocator,
    });
}

fn _max_backward(T: type, self: NDTensor(T), _: std.mem.Allocator) !void {
    if (self.children) |children| {
        const child = children[0];
        const max_val = self.data.data[0];
        for (child.data.data, 0..) |val, i| {
            if (val == max_val) {
                child.grad.?.data[i] += self.grad.?.data[0];
            }
        }
    }
}

/// Element-wise exponential. COM.
pub fn exp(T: type, self: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    return try NDTensor(T).createDependent(.{
        .data = try self.data.exp(allocator),
        .op = .EXP,
        .children = &[_]*const NDTensor(T){self},
        .requires_grad = self.requires_grad,
        ._backward = _exp_backward,
        .allocator = allocator,
    });
}

fn _exp_backward(T: type, self: NDTensor(T), _: std.mem.Allocator) !void {
    if (self.children) |children| {
        const child = children[0];
        for (self.data.data, self.grad.?.data, 0..) |exp_val, grad_val, i| {
            child.grad.?.data[i] += exp_val * grad_val;
        }
    }
}

/// Sum of all elements in the tensor. COM.
pub fn sum(T: type, self: *const NDTensor(T), allocator: std.mem.Allocator) !*NDTensor(T) {
    return try NDTensor(T).createDependent(.{
        .data = try self.data.sum(allocator),
        .op = .SUM,
        .children = &[_]*const NDTensor(T){self},
        .requires_grad = self.requires_grad,
        ._backward = _sum_backward,
        .allocator = allocator,
    });
}

fn _sum_backward(T: type, self: NDTensor(T), _: std.mem.Allocator) !void {
    if (self.children) |children| {
        const child = children[0];
        _ = try child.grad.?._add(self.grad.?);
    }
}
