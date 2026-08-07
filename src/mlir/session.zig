//! MLIR registry and context lifecycle.
//!
//! A session registers the `func` dialect. Dialect-specific code loads any
//!  additional dialects it requires.
//!
//! ## Comptime dialect constraint
//!
//! `load_dialect` requires a comptime-known dialect name because
//!  `DialectHandle.from_string` resolves a linked C symbol at compile time.
const mlir = @import("../c/mlir/mlir.zig");

/// MLIR registry and context pair released with `deinit`.
pub const Session = struct {
    registry: mlir.Registry,
    ctx: mlir.Context,

    /// Create a baseline context with the `func` dialect.
    pub fn init() mlir.Error!Session {
        var registry = try mlir.Registry.init();
        errdefer registry.deinit();

        const func_handle = mlir.DialectHandle.from_string("func");
        func_handle.insert_dialect(registry);

        var ctx = try mlir.Context.init_with_registry(registry, false);
        errdefer ctx.deinit();
        ctx.allow_unregistered_dialects(false);

        func_handle.register_dialect(ctx);
        _ = func_handle.load_dialect(ctx);

        return .{
            .registry = registry,
            .ctx = ctx,
        };
    }

    /// Register and load one linked dialect.
    ///
    /// The dialect name must be comptime-known.
    pub fn load_dialect(self: Session, comptime name: [:0]const u8) void {
        const handle = mlir.DialectHandle.from_string(name);
        handle.register_dialect(self.ctx);
        _ = handle.load_dialect(self.ctx);
    }

    pub fn deinit(self: *Session) void {
        self.ctx.deinit();
        self.registry.deinit();
    }
};
