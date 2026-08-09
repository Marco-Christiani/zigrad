//! PR region outlining.
const pr = @import("pr.zig");

/// Region annotation requesting PR function outlining.
pub const annotation_name = "zigrad.outline";

/// Unit-valued outline request for region builders.
pub const annotation: pr.Annotation = .{
    .name = annotation_name,
    .value = .unit,
};

pub const AnnotationError = error{InvalidOutlineAnnotation};

/// Return whether a region requests outlining.
pub fn is_requested(region: pr.Region) AnnotationError!bool {
    const found = region.find_annotation(annotation_name) orelse return false;
    return switch (found.value) {
        .unit => true,
        .boolean => |value| value,
        else => error.InvalidOutlineAnnotation,
    };
}

test is_requested {
    const requested = pr.Region{
        .id = 0,
        .name = "requested",
        .annotations = &.{annotation},
        .op_ids = &.{},
    };
    try @import("std").testing.expect(try is_requested(requested));
}
