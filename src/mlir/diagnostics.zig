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
    missing_libmlir: bool,
    missing_libllvm: bool,
    symbol_in_process: bool,
    symbol_in_libmlir: bool,
};

pub fn checkStablehloDialectSupport() DialectRegistrationDiagnostic {
    const symbol_name = "mlirGetDialectHandle__stablehlo__";
    const symbol_in_process = c.dlsym(c.RTLD_DEFAULT, symbol_name) != null;

    const lib_mlir = "libMLIR.so";
    const lib_llvm = "libLLVM.so";

    const mlir_handle = c.dlopen(lib_mlir, c.RTLD_LAZY | c.RTLD_LOCAL);
    const llvm_handle = c.dlopen(lib_llvm, c.RTLD_LAZY | c.RTLD_LOCAL);

    const missing_libmlir = (mlir_handle == null);
    const missing_libllvm = (llvm_handle == null);

    var symbol_in_libmlir = false;
    if (mlir_handle != null) {
        symbol_in_libmlir = c.dlsym(mlir_handle, symbol_name) != null;
    }

    if (mlir_handle != null) {
        _ = c.dlclose(mlir_handle);
    }
    if (llvm_handle != null) {
        _ = c.dlclose(llvm_handle);
    }

    const issue = if (symbol_in_process)
        DialectRegistrationIssue.none
    else if (missing_libmlir or missing_libllvm)
        DialectRegistrationIssue.missing_dso
    else if (symbol_in_libmlir)
        DialectRegistrationIssue.load_order
    else
        DialectRegistrationIssue.missing_symbol;

    return .{
        .issue = issue,
        .missing_libmlir = missing_libmlir,
        .missing_libllvm = missing_libllvm,
        .symbol_in_process = symbol_in_process,
        .symbol_in_libmlir = symbol_in_libmlir,
    };
}
