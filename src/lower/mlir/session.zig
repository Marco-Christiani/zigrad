//! MLIR session: registry + context lifecycle init.
//!
//! `MlirSession.init` provides a baseline MLIR context with only the `func`
//!  and zigrad and dialects registered. Dialect-specific code (StableHLO,
//!  linalg, etc.) calls `load_dialect` after init to register what it needs.
//!
//! ## Comptime dialect constraint
//!
//! `load_dialect` requires a comptime-known dialect name because
//!  `DialectHandle.from_string` resolves to a C symbol
//!  (`mlirGetDialectHandle__<name>__`) at compile time. The dialect must be
//!  linked into the binary. Truly dynamic dialect loading (from a .so at
//!  runtime) would require a runtime `DialectHandle` path.
const std = @import("std");

const mlir = @import("../../c/mlir/mlir.zig");

const log = std.log.scoped(.@"zg/mlir_context");

/// Owns an MLIR registry + context pair. Call `deinit` when done.
///
/// Baseline init registers only the `func` dialect and zigrad MLIR
///  extensions. Callers add dialect-specific registrations via
///  `load_dialect` before using the context.
pub const MlirSession = struct {
    registry: mlir.Registry,
    ctx: mlir.Context,

    /// Baseline MLIR init: `func` dialect + zigrad extensions only.
    ///
    /// Callers must use `load_dialect` for each dialect they need before
    ///  lowering or parsing MLIR that contains dialect-specific ops.
    /// TODO: consider a singleton?
    pub fn init() mlir.Error!MlirSession {
        var registry = try mlir.Registry.init();
        errdefer registry.deinit();

        const func_handle = mlir.DialectHandle.from_string("func");
        func_handle.insert_dialect(registry);

        var ctx = try mlir.Context.init_with_registry(registry, false);
        errdefer ctx.deinit();
        ctx.allow_unregistered_dialects(false);

        mlir.register_zigrad_extensions(ctx) catch |e| {
            log.err("set ZG_MLIR_SHIM_PATH or provide ZG_EXTERNAL_SDK_ROOT with lib/libzigrad_mlir_ext.so", .{});
            return e;
        };

        func_handle.register_dialect(ctx);
        _ = func_handle.load_dialect(ctx);

        return .{
            .registry = registry,
            .ctx = ctx,
        };
    }

    /// Register and load a dialect into this session's context.
    ///
    /// The dialect name must be comptime-known (resolves to a linked C symbol).
    /// The dialect library must be linked into the binary.
    pub fn load_dialect(self: MlirSession, comptime name: [:0]const u8) void {
        const handle = mlir.DialectHandle.from_string(name);
        handle.register_dialect(self.ctx);
        _ = handle.load_dialect(self.ctx);
    }

    pub fn deinit(self: *MlirSession) void {
        self.ctx.deinit();
        self.registry.deinit();
    }
};
