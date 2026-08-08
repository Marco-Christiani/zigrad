/// Return the concrete operation type represented by `T`.
pub fn type_of(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |pointer| pointer.child,
        else => T,
    };
}

/// Check the structural operation contract.
pub fn validate(comptime Operation: type) void {
    if (!@hasDecl(Operation, "Input")) {
        @compileError(@typeName(Operation) ++ " must declare Input");
    }
    if (!@hasDecl(Operation, "Output")) {
        @compileError(@typeName(Operation) ++ " must declare Output");
    }
    if (!@hasDecl(Operation, "run")) {
        @compileError(@typeName(Operation) ++ " must provide run");
    }
}
