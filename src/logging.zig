//! This is a copy of the standard library logging source code.
//!
//! This version makes logs opt-in and does not assume a default
//! level based on build mode. Scopes are also limited to modules
//! within zigrad.
//!
const std = @import("std");
const builtin = @import("builtin");
const zg = @import("zigrad.zig");

/// Examples of supported modules in Zigrad:
/// - zg_ndtensor
/// - zg_ndarray
/// - zg_block_pool
/// - zg_caching_allocator
///
const Scope = @Type(.enum_literal);

pub const Level = std.log.Level;

pub const ScopeLevel = struct {
    scope: Scope,
    level: Level,
};

pub const LoggingFunction = fn (
    comptime message_level: Level,
    comptime scope: Scope,
    comptime format: []const u8,
    args: anytype,
) void;

const level = zg.settings.logging.level;
const scope_levels = zg.settings.logging.scopes;
const callback = zg.settings.logging.callback orelse default_callback;

fn log(
    comptime message_level: Level,
    comptime scope: Scope,
    comptime format: []const u8,
    args: anytype,
) void {
    if (comptime !logEnabled(message_level, scope))
        return;

    callback(message_level, scope, format, args);
}

/// Determine if a specific log message level and scope combination are enabled for logging.
pub fn logEnabled(comptime message_level: Level, comptime scope: Scope) bool {
    return inline for (scope_levels) |scope_level| {
        if (scope_level.scope == scope) break @intFromEnum(message_level) <= @intFromEnum(scope_level.level);
    } else false;
}

/// The default implementation for the log function, custom log functions may
/// forward log messages to this function.
pub fn default_callback(
    comptime message_level: Level,
    comptime scope: Scope,
    comptime format: []const u8,
    args: anytype,
) void {
    const level_txt = comptime message_level.asText();
    const prefix2 = "(" ++ @tagName(scope) ++ "): ";
    const stderr = std.io.getStdErr().writer();
    var bw = std.io.bufferedWriter(stderr);
    const writer = bw.writer();

    std.debug.lockStdErr();
    defer std.debug.unlockStdErr();
    nosuspend {
        writer.print(level_txt ++ prefix2 ++ format ++ "\n", args) catch return;
        bw.flush() catch return;
    }
}

/// Returns a scoped logging namespace that logs all messages using the scope
/// provided here.
pub fn scoped(comptime scope: Scope) type {
    return struct {
        /// Log an error message. This log level is intended to be used
        /// when something has gone wrong. This might be recoverable or might
        /// be followed by the program exiting.
        pub fn err(
            comptime format: []const u8,
            args: anytype,
        ) void {
            @branchHint(.cold);
            log(.err, scope, format, args);
        }

        /// Log a warning message. This log level is intended to be used if
        /// it is uncertain whether something has gone wrong or not, but the
        /// circumstances would be worth investigating.
        pub fn warn(
            comptime format: []const u8,
            args: anytype,
        ) void {
            log(.warn, scope, format, args);
        }

        /// Log an info message. This log level is intended to be used for
        /// general messages about the state of the program.
        pub fn info(
            comptime format: []const u8,
            args: anytype,
        ) void {
            log(.info, scope, format, args);
        }

        /// Log a debug message. This log level is intended to be used for
        /// messages which are only useful for debugging.
        pub fn debug(
            comptime format: []const u8,
            args: anytype,
        ) void {
            log(.debug, scope, format, args);
        }
    };
}
