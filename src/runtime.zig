//! Process-wide runtime carriers (allocator, io, environ).
//!
//! Built once in `main` from `std.process.Init`, then threaded through
//!  CLI dispatch and into any subsystem that needs I/O, env access, or
//!  allocation. Pure subsystems (PR, lower, kernel contracts) do not see this.
const std = @import("std");

pub const RuntimeEnv = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    environ: *std.process.Environ.Map,

    pub fn from_init(init: std.process.Init) RuntimeEnv {
        return .{
            .allocator = init.gpa,
            .io = init.io,
            .environ = init.environ_map,
        };
    }
};
