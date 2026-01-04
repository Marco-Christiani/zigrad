const std = @import("std");

const c = @cImport({
    @cInclude("dlfcn.h");
});

pub const DialectRegistrationIssue = enum {
    none,
    missing_dso,
    missing_symbol,
    load_order,
};

pub const DialectRegistrationDiagnostic = struct {
    issue: DialectRegistrationIssue,
    /// null => unknown (non-invasive mode)
    missing_libmlir_c: ?bool,
    /// null => unknown (non-invasive mode)
    missing_libstablehlo_capi: ?bool,
    symbol_in_process: bool,
    /// null => unknown (non-invasive mode)
    symbol_in_libstablehlo_capi: ?bool,
};

/// Non-invasive probe that does do not load DSOs.
///
/// Checks whether the symbol is already present in the process
///  via DT_NEEDED + RUNPATH resolution.
pub fn checkStablehloDialectSupport() DialectRegistrationDiagnostic {
    const symbol_name = "mlirGetDialectHandle__stablehlo__";
    const symbol_in_process = c.dlsym(c.RTLD_DEFAULT, symbol_name) != null;

    return .{
        .issue = if (symbol_in_process) .none else .missing_symbol,
        .missing_libmlir_c = null,
        .missing_libstablehlo_capi = null,
        .symbol_in_process = symbol_in_process,
        .symbol_in_libstablehlo_capi = null,
    };
}

/// Invasive probe that attempts to dlopen relevant DSOs to narrow down failure modes.
///
/// Warning: dlopen is not read-only; it can load new DSOs, run initializers, and affect later symbol resolution.
pub fn checkStablehloDialectSupportInvasive() DialectRegistrationDiagnostic {
    const symbol_name = "mlirGetDialectHandle__stablehlo__";
    const symbol_in_process = c.dlsym(c.RTLD_DEFAULT, symbol_name) != null;

    // StableHLO dialect handle is expected to be provided by libStablehloCAPI.so (shared)
    // MLIR C API boundary is provided by libMLIR-C.so
    const lib_mlir_c = "libMLIR-C.so";
    const lib_stablehlo_capi = "libStablehloCAPI.so";

    const mlir_c_handle = c.dlopen(lib_mlir_c, c.RTLD_NOW | c.RTLD_LOCAL);
    const stablehlo_handle = c.dlopen(lib_stablehlo_capi, c.RTLD_NOW | c.RTLD_LOCAL);

    const missing_libmlir_c = (mlir_c_handle == null);
    const missing_libstablehlo_capi = (stablehlo_handle == null);

    var symbol_in_libstablehlo_capi = false;
    if (stablehlo_handle != null) {
        symbol_in_libstablehlo_capi = c.dlsym(stablehlo_handle, symbol_name) != null;
    }

    if (mlir_c_handle != null) _ = c.dlclose(mlir_c_handle);
    if (stablehlo_handle != null) _ = c.dlclose(stablehlo_handle);

    const issue = if (symbol_in_process)
        DialectRegistrationIssue.none
    else if (missing_libmlir_c or missing_libstablehlo_capi)
        DialectRegistrationIssue.missing_dso
    else if (symbol_in_libstablehlo_capi)
        DialectRegistrationIssue.load_order
    else
        DialectRegistrationIssue.missing_symbol;

    return .{
        .issue = issue,
        .missing_libmlir_c = missing_libmlir_c,
        .missing_libstablehlo_capi = missing_libstablehlo_capi,
        .symbol_in_process = symbol_in_process,
        .symbol_in_libstablehlo_capi = symbol_in_libstablehlo_capi,
    };
}
