const std = @import("std");
const log = std.log.scoped(.@"zg/loop_timer");

/// Convenience timer for timing events in loops.
///
/// Provides timing bookkeeping and formatted per-step logging,
///  does NOT manage barriers, that is the user's responsbility
///  and a requirement for valid results.
///
/// Usage:
///
/// ```zig
/// var timer = LoopTimer{ .label = "train" };
/// for (0..steps) |_| {
///     try timer.start_step();
///     // ... work ...
///     // (barrier)
///     timer.mark("dispatch");
///     // ... more work ...
///     // (barrier)
///     timer.mark("exec");
///     timer.end_step(loss);
/// }
/// ```
pub const LoopTimer = struct {
    const max_phases = 8;

    const Phase = struct { name: []const u8, ns: u64 };

    label: []const u8,
    quiet: bool = false,
    total_ns: u64 = 0,
    step_count: usize = 0,
    // per-step scratch
    timer: std.time.Timer = undefined,
    phases: [max_phases]Phase = undefined,
    phase_count: usize = 0,

    /// Begin timing a new step. Resets phase counter and starts the clock.
    pub fn start_step(self: *LoopTimer) !void {
        self.timer = try std.time.Timer.start();
        self.phase_count = 0;
    }

    /// Record elapsed time since the last `mark` (or `start_step`) as a
    /// named phase. Up to `max_phases` (8) marks per step.
    pub fn mark(self: *LoopTimer, name: []const u8) void {
        std.debug.assert(self.phase_count < max_phases);
        self.phases[self.phase_count] = .{ .name = name, .ns = self.timer.lap() };
        self.phase_count += 1;
    }

    /// Finalize the current step. Accumulates total time, increments step
    ///  count, and (unless `quiet`) logs a line with per-phase and total
    ///  durations plus the optional loss value.
    pub fn end_step(self: *LoopTimer, loss: ?f32) void {
        var step_ns: u64 = 0;
        for (self.phases[0..self.phase_count]) |p| step_ns += p.ns;
        self.total_ns += step_ns;
        if (!self.quiet) self.log_step(step_ns, loss);
        self.step_count += 1;
    }

    /// Average step duration in milliseconds across all completed steps.
    /// Returns 0 if no steps have been recorded.
    pub fn avg_ms(self: LoopTimer) f64 {
        if (self.step_count == 0) return 0;
        return ns_to_ms(self.total_ns) / @as(f64, @floatFromInt(self.step_count));
    }

    fn log_step(self: *const LoopTimer, step_ns: u64, loss: ?f32) void {
        var buf: [512]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const w = fbs.writer();
        w.print("{s} step {d}:", .{ self.label, self.step_count }) catch return;
        if (loss) |l| w.print(" loss={d:.6}", .{l}) catch return;
        for (self.phases[0..self.phase_count]) |p| {
            w.print(" {s}={d:.3}ms", .{ p.name, ns_to_ms(p.ns) }) catch return;
        }
        w.print(" total={d:.3}ms", .{ns_to_ms(step_ns)}) catch return;
        log.info("{s}", .{fbs.getWritten()});
    }

    fn ns_to_ms(ns: u64) f64 {
        return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
    }
};
