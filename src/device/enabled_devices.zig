const std = @import("std");
const build_options = @import("build_options");
const HostDevice = @import("host_device.zig");
const CudaDevice = @import("cuda_device.zig");

/// Build Enabled Device Pointers
///
/// This corresponds to build_options - to enable devices, pass
/// the related flag to the module. Each enabled device adds a
/// field to the internal DeviceReference.ptrs field. To add a
/// new device, add new flag to the build and a new block to
/// this function. The field type should be a pointer to the
/// device type.
pub const EnabledDevicePointers: type = blk: {
    // assuming we'll never have more than 10 devices
    var union_fields: [10]std.builtin.Type.UnionField = undefined;
    var enum_fields: [10]std.builtin.Type.EnumField = undefined;
    var size: usize = 1;

    // HostDevice is always enabled.
    union_fields[0] = .{
        .type = *HostDevice,
        .name = "host",
        .alignment = @alignOf(*HostDevice),
    };
    enum_fields[0] = .{
        .name = "host",
        .value = 0,
    };

    if (build_options.enable_cuda) {
        union_fields[size] = .{
            .type = *CudaDevice,
            .name = "cuda",
            .alignment = @alignOf(*CudaDevice),
        };
        enum_fields[size] = .{
            .name = "cuda",
            .value = 1,
        };
        size += 1;
    }

    const Tag = @Type(.{
        .@"enum" = std.builtin.Type.Enum{
            .tag_type = u8,
            .fields = enum_fields[0..size],
            .decls = &.{},
            .is_exhaustive = true,
        },
    });

    break :blk @Type(.{
        .@"union" = std.builtin.Type.Union{
            .layout = .auto,
            .tag_type = Tag,
            .fields = union_fields[0..size],
            .decls = &.{},
        },
    });
};
