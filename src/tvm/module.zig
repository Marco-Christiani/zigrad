//! Loading and management of tuned TVM modules.
//!
//! After MetaSchedule tuning produces candidate .so files and a
//! tuning_record.json, this module parses the records, ranks candidates
//! by measured runtime, and loads the best one.

const std = @import("std");
const api = @import("../c/tvm/api.zig");
const RuntimeModule = @import("../c/tvm/runtime.zig").RuntimeModule;
const Value = api.Value;
const TargetKind = @import("../c/tvm/tir.zig").TargetKind;

const log = std.log.scoped(.@"zg/tvm_loader");

/// Options for loading a tuned module.
pub const LoadOpts = struct {
    /// Base directory containing tuning artifacts (tuning_record.json + candidate .so files).
    work_dir: []const u8 = "artifacts/tvm_cache",
    /// Target to load tuned modules for. Determines the per-target subdirectory.
    target_kind: TargetKind = .cpu,
};

/// Result of loading a tuned module.
pub const TunedModule = struct {
    module: RuntimeModule,
    main_func: Value,
    best_candidate: usize,
    best_time_us: f64,

    pub fn deinit(self: *TunedModule) void {
        self.main_func.decref();
        self.module.deinit();
        self.* = undefined;
    }

    /// Invoke the tuned main function with the given arguments.
    pub fn invoke(self: TunedModule, allocator: std.mem.Allocator, args: []const Value) !void {
        const func_handle = self.main_func.as_object() orelse return error.TvmCallFailed;
        _ = try api.call_handle(allocator, func_handle, args);
    }
};

/// Load the best tuned module from a previous tuning run.
///
/// Parses tuning_record.json to find the fastest candidate, then loads
/// the corresponding .so file and returns a handle to the main function.
pub fn load(allocator: std.mem.Allocator, opts: LoadOpts) !TunedModule {
    const record_path = try std.fmt.allocPrint(allocator, "{s}/tuning_record.json", .{opts.work_dir});
    defer allocator.free(record_path);

    const candidates = try find_ranked_candidates(allocator, record_path, opts.work_dir);
    defer allocator.free(candidates);

    const best = candidates[0];
    var tuned = try load_candidate(allocator, opts.work_dir, best.idx);
    tuned.best_time_us = best.time_secs * 1e6;

    log.info("loaded tuned module (candidate {d}, {d:.2} us)", .{ best.idx, tuned.best_time_us });
    return tuned;
}

// ============================================================================
// Private helpers
// ============================================================================

const RankedCandidate = struct {
    idx: usize,
    time_secs: f64,
};

/// Parse tuning_record.json and return candidates ranked by speed (fastest first).
///
/// Only includes candidates whose .so file exists on disk.
fn find_ranked_candidates(allocator: std.mem.Allocator, record_path: []const u8, work_dir: []const u8) ![]RankedCandidate {
    const file = std.fs.cwd().openFile(record_path, .{}) catch |err| {
        log.err("failed to open tuning records at {s}: {s}", .{ record_path, @errorName(err) });
        return error.NoTuningRecords;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    if (file_size == 0) {
        log.err("empty tuning records file: {s}", .{record_path});
        return error.NoTuningRecords;
    }
    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);
    const bytes_read = try file.readAll(contents);

    var candidates = std.ArrayList(RankedCandidate).empty;
    defer candidates.deinit(allocator);
    var line_num: usize = 0;

    var lines = std.mem.splitScalar(u8, contents[0..bytes_read], '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        // Extract run_secs with pattern matching instead of a full JSON parse.
        // Format: [workload_id, [[trace, decisions], [run_secs], target, args]]
        // Look for pattern "]],[" followed by a float.
        const run_secs = parse_run_secs(line);

        if (run_secs) |t| {
            const so_path = try std.fmt.allocPrint(allocator, "{s}/candidate_{d}.so", .{ work_dir, line_num });
            defer allocator.free(so_path);

            std.fs.cwd().access(so_path, .{}) catch {
                log.debug("skipping candidate {d}: .so not found", .{line_num});
                line_num += 1;
                continue;
            };

            try candidates.append(allocator, .{ .idx = line_num, .time_secs = t });
        } else {
            log.warn("could not parse run_secs from record {d}", .{line_num});
        }

        line_num += 1;
    }

    if (candidates.items.len == 0) {
        log.err("no valid tuning records found in {s}", .{record_path});
        return error.NoTuningRecords;
    }

    const items = try candidates.toOwnedSlice(allocator);
    std.mem.sort(RankedCandidate, items, {}, struct {
        fn lessThan(_: void, a: RankedCandidate, b: RankedCandidate) bool {
            return a.time_secs < b.time_secs;
        }
    }.lessThan);

    log.info("found {d} candidates, fastest: {d} ({d:.2} us)", .{
        items.len, items[0].idx, items[0].time_secs * 1e6,
    });

    return items;
}

/// Extract run_secs from a tuning_record.json line via pattern matching.
///
/// Looks for `]],[` followed by a float (the run_secs array element).
fn parse_run_secs(line: []const u8) ?f64 {
    var i: usize = 0;
    while (i + 10 < line.len) : (i += 1) {
        if (i + 4 < line.len and
            line[i] == ']' and line[i + 1] == ']' and
            line[i + 2] == ',' and line[i + 3] == '[')
        {
            const start = i + 4;
            if (start < line.len and std.ascii.isDigit(line[start])) {
                var end = start;
                while (end < line.len and line[end] != ']') : (end += 1) {}
                if (end > start) {
                    const num_str = line[start..end];
                    if (std.mem.indexOfScalar(u8, num_str, 'e') != null or
                        std.mem.indexOfScalar(u8, num_str, '.') != null)
                    {
                        const val = std.fmt.parseFloat(f64, num_str) catch null;
                        if (val != null) return val;
                    }
                }
            }
        }
    }
    return null;
}

/// Load a specific candidate module by index.
fn load_candidate(allocator: std.mem.Allocator, work_dir: []const u8, candidate_idx: usize) !TunedModule {
    const so_path = try std.fmt.allocPrint(allocator, "{s}/candidate_{d}.so", .{ work_dir, candidate_idx });
    defer allocator.free(so_path);

    log.info("loading tuned module: {s}", .{so_path});

    var module = try RuntimeModule.load_from_file(allocator, so_path);
    errdefer module.deinit();

    const main_func = try module.get_function(allocator, "main", true);

    return .{
        .module = module,
        .main_func = main_func,
        .best_candidate = candidate_idx,
        .best_time_us = 0, // caller fills this in
    };
}
