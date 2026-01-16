const std = @import("std");

var unregistered_dialects = std.atomic.Value(bool).init(false);

pub fn markUnregisteredDialects() void {
    unregistered_dialects.store(true, .seq_cst);
}

pub fn isUnregisteredDialects() bool {
    return unregistered_dialects.load(.seq_cst);
}
