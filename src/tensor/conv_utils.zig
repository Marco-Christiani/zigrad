const std = @import("std");
const NDArray = @import("zarray.zig").NDArray;

pub fn im2col(comptime T: type, input: *NDArray(T), kernel_size: usize, stride: usize, padding: usize, allocator: std.mem.Allocator) !*NDArray(T) {
    const batch_size = input.shape.shape[0];
    const channels = input.shape.shape[1];
    const height = input.shape.shape[2];
    const width = input.shape.shape[3];

    const output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const output_width = (width + 2 * padding - kernel_size) / stride + 1;

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    var col = try NDArray(T).init(try allocator.alloc(T, batch_size * col_height * col_width), &[_]usize{ batch_size, col_height, col_width }, allocator);

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = c % kernel_size;
            const h_offset = (c / kernel_size) % kernel_size;
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = h * stride - padding + h_offset;
                    const w_pad = w * stride - padding + w_offset;
                    if (h_pad >= 0 and h_pad < height and w_pad >= 0 and w_pad < width) {
                        const input_index = b * channels * height * width + c_im * height * width + h_pad * width + w_pad;
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        col.data[col_index] = input.data[input_index];
                    } else {
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        col.data[col_index] = 0;
                    }
                }
            }
        }
    }

    return col;
}

pub fn col2im(comptime T: type, col: *NDArray(T), input_shape: []const usize, kernel_size: usize, stride: usize, padding: usize, allocator: std.mem.Allocator) !*NDArray(T) {
    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const height = input_shape[2];
    const width = input_shape[3];

    const output_height = (height + 2 * padding - kernel_size) / stride + 1;
    const output_width = (width + 2 * padding - kernel_size) / stride + 1;

    var im = try NDArray(T).init(try allocator.alloc(T, batch_size * channels * height * width), input_shape, allocator);
    @memset(im.data, 0);

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = c % kernel_size;
            const h_offset = (c / kernel_size) % kernel_size;
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = h * stride - padding + h_offset;
                    const w_pad = w * stride - padding + w_offset;
                    if (h_pad >= 0 and h_pad < height and w_pad >= 0 and w_pad < width) {
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        const im_index = b * channels * height * width + c_im * height * width + h_pad * width + w_pad;
                        im.data[im_index] += col.data[col_index];
                    }
                }
            }
        }
    }

    return im;
}
