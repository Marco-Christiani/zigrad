//! Decode CIFAR-10 binary records into the model's NHWC representation.

const std = @import("std");

pub const image_size: usize = 32;
pub const channel_count: usize = 3;
pub const class_count: usize = 10;
pub const record_size: usize = 1 + channel_count * image_size * image_size;
pub const records_per_file: usize = 10_000;
pub const training_file_count: usize = 5;

/// Validate record alignment and class labels.
pub fn validate(records: []const u8) !void {
    if (records.len == 0 or records.len % record_size != 0) return error.InvalidDataset;
    var offset: usize = 0;
    while (offset < records.len) : (offset += record_size) {
        if (records[offset] >= class_count) return error.InvalidDataset;
    }
}

/// Decode `count` records into a contiguous NHWC batch, wrapping at the end.
///
/// `records` must be nonempty and accepted by `validate`.
pub fn decode_images(output: []f32, records: []const u8, start_record: usize, count: usize) void {
    const record_count = records.len / record_size;
    std.debug.assert(output.len == count * image_size * image_size * channel_count);
    for (0..count) |batch_index| {
        const record = records[((start_record + batch_index) % record_count) * record_size ..][0..record_size];
        for (0..image_size) |row| for (0..image_size) |column| for (0..channel_count) |channel| {
            const planar_index = channel * image_size * image_size + row * image_size + column;
            const nhwc_index = ((batch_index * image_size + row) * image_size + column) * channel_count + channel;
            output[nhwc_index] = @as(f32, @floatFromInt(record[1 + planar_index])) / 255.0;
        };
    }
}

/// Write one-hot labels for `count` records, wrapping at the end.
///
/// `records` must be nonempty and accepted by `validate`.
pub fn decode_labels(output: []f32, records: []const u8, start_record: usize, count: usize) void {
    const record_count = records.len / record_size;
    std.debug.assert(output.len == count * class_count);
    @memset(output, 0);
    for (0..count) |batch_index| {
        const record_index = (start_record + batch_index) % record_count;
        const label = records[record_index * record_size];
        output[batch_index * class_count + label] = 1;
    }
}

test validate {
    var record: [record_size]u8 = @splat(0);
    try validate(&record);
    record[0] = class_count;
    try std.testing.expectError(error.InvalidDataset, validate(&record));
}
