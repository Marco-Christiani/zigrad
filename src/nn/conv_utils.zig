const std = @import("std");
const zg = @import("../zigrad.zig");
const NDArray = zg.NDArray;
const DeviceReference = zg.DeviceReference;

pub fn im2col(comptime T: type, input: NDArray(T), kernel_size: usize, stride: usize, padding: usize, dilation: usize, device: DeviceReference) !*NDArray(T) {
    const batch_size = input.shape.shape[0];
    const channels = input.shape.shape[1];
    const height = input.shape.shape[2];
    const width = input.shape.shape[3];

    const output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    var col = try NDArray(T).empty(&[_]usize{ batch_size, col_height, col_width }, device);

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = @as(i64, @intCast(c % kernel_size));
            const h_offset = @as(i64, @intCast((c / kernel_size) % kernel_size));
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = @as(i64, @intCast(h * stride)) - @as(i64, @intCast(padding)) + h_offset * @as(i64, @intCast(dilation));
                    const w_pad = @as(i64, @intCast(w * stride)) - @as(i64, @intCast(padding)) + w_offset * @as(i64, @intCast(dilation));
                    if (h_pad >= 0 and h_pad < @as(i64, @intCast(height)) and w_pad >= 0 and w_pad < @as(i64, @intCast(width))) {
                        const input_index = b * channels * height * width + c_im * height * width + @as(usize, @intCast(h_pad)) * width + @as(usize, @intCast(w_pad));
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

pub fn col2im(comptime T: type, col: NDArray(T), input_shape: []const usize, kernel_size: usize, stride: usize, padding: usize, dilation: usize, device: DeviceReference) !*NDArray(T) {
    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const height = input_shape[2];
    const width = input_shape[3];

    const output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    var im = try NDArray(T).empty(input_shape, device);
    @memset(im.data, 0);

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = @as(i64, @intCast(c % kernel_size));
            const h_offset = @as(i64, @intCast((c / kernel_size) % kernel_size));
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = @as(i64, @intCast(h * stride)) - @as(i64, @intCast(padding)) + h_offset * @as(i64, @intCast(dilation));
                    const w_pad = @as(i64, @intCast(w * stride)) - @as(i64, @intCast(padding)) + w_offset * @as(i64, @intCast(dilation));
                    if (h_pad >= 0 and h_pad < @as(i64, @intCast(height)) and w_pad >= 0 and w_pad < @as(i64, @intCast(width))) {
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        const im_index = b * channels * height * width + c_im * height * width + @as(usize, @intCast(h_pad)) * width + @as(usize, @intCast(w_pad));
                        im.data[im_index] += col.data[col_index];
                    }
                }
            }
        }
    }

    return im;
}

test "im2col col2im" {
    var cpu = zg.device.HostDevice.init(std.testing.allocator);
    defer cpu.deinit();
    const device = cpu.reference();

    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const input_shape = [_]usize{ 1, 1, 4, 4 };
    const kernel_size: usize = 3;
    const stride: usize = 1;
    const padding: usize = 1;
    const dilation: usize = 1;

    var input = try NDArray(f32).init(&input_data, &input_shape, device);
    defer input.deinit(device);

    // im2col
    var col = try im2col(f32, input.*, kernel_size, stride, padding, dilation, device);
    defer col.deinit(device);

    const expected_col_data = [_]f32{
        0, 0, 0, 0, 0,  1,  2,  3,  0,  5,  6,  7,  0,  9,  10, 11,
        0, 0, 0, 0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        0, 0, 0, 0, 2,  3,  4,  0,  6,  7,  8,  0,  10, 11, 12, 0,
        0, 1, 2, 3, 0,  5,  6,  7,  0,  9,  10, 11, 0,  13, 14, 15,
        1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        2, 3, 4, 0, 6,  7,  8,  0,  10, 11, 12, 0,  14, 15, 16, 0,
        0, 5, 6, 7, 0,  9,  10, 11, 0,  13, 14, 15, 0,  0,  0,  0,
        5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16, 0,  0,  0,  0,
        6, 7, 8, 0, 10, 11, 12, 0,  14, 15, 16, 0,  0,  0,  0,  0,
    };

    std.debug.print("\ncol data:\n", .{});
    col.print();

    try std.testing.expectEqualSlices(f32, &expected_col_data, col.data);

    // col2im
    var im = try col2im(f32, col.*, &input_shape, kernel_size, stride, padding, dilation, device);
    defer im.deinit(device);

    const exp_im_data = [_]f32{ 4, 12, 18, 16, 30, 54, 63, 48, 54, 90, 99, 72, 52, 84, 90, 64 };
    try std.testing.expectEqualSlices(f32, &exp_im_data, im.data);

    std.debug.print("\nim data:\n", .{});
    im.print();
}
