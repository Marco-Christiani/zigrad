//! Loading and management of tuned TVM modules.
//!
//! After MetaSchedule tuning produces candidate .so files and a
//! tuning_record.json, this module parses the records, ranks candidates
//! by measured runtime, and loads the best one.
//!
//! Also maintains a cache index for stable lookup by kernel key.

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

/// Cache entry for a tuned kernel artifact.
pub const CacheEntry = struct {
    key: []const u8,
    target_kind: TargetKind,
    artifact_path: []const u8,
    best_time_us: f64,
};

const CacheIndex = struct {
    version: u32 = 1,
    entries: []CacheEntry = &.{},
};

pub fn cache_lookup(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    key: []const u8,
    target_kind: TargetKind,
) !?CacheEntry {
    var index = try read_cache_index(allocator, base_dir);
    defer deinit_cache_index(allocator, &index);

    for (index.entries) |entry| {
        if (entry.target_kind != target_kind) continue;
        if (std.mem.eql(u8, entry.key, key)) {
            return .{
                .key = try allocator.dupe(u8, entry.key),
                .target_kind = entry.target_kind,
                .artifact_path = try allocator.dupe(u8, entry.artifact_path),
                .best_time_us = entry.best_time_us,
            };
        }
    }
    return null;
}

pub fn cache_update(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    entry: CacheEntry,
) !void {
    var index = try read_cache_index(allocator, base_dir);
    defer deinit_cache_index(allocator, &index);

    for (index.entries) |*existing| {
        if (existing.target_kind == entry.target_kind and std.mem.eql(u8, existing.key, entry.key)) {
            allocator.free(existing.key);
            allocator.free(existing.artifact_path);
            existing.key = try allocator.dupe(u8, entry.key);
            existing.artifact_path = try allocator.dupe(u8, entry.artifact_path);
            existing.best_time_us = entry.best_time_us;
            return write_cache_index(allocator, base_dir, index);
        }
    }

    var entries = try allocator.alloc(CacheEntry, index.entries.len + 1);
    @memcpy(entries[0..index.entries.len], index.entries);
    entries[index.entries.len] = .{
        .key = try allocator.dupe(u8, entry.key),
        .target_kind = entry.target_kind,
        .artifact_path = try allocator.dupe(u8, entry.artifact_path),
        .best_time_us = entry.best_time_us,
    };
    allocator.free(index.entries);
    index.entries = entries;
    return write_cache_index(allocator, base_dir, index);
}

pub fn best_candidate_from_records(allocator: std.mem.Allocator, work_dir: []const u8) !RankedCandidate {
    const record_path = try std.fmt.allocPrint(allocator, "{s}/tuning_record.json", .{work_dir});
    defer allocator.free(record_path);

    const candidates = try find_ranked_candidates(allocator, record_path, work_dir);
    defer allocator.free(candidates);
    return candidates[0];
}

pub fn stable_artifact_path(allocator: std.mem.Allocator, work_dir: []const u8, key: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/kernel_{s}.so", .{ work_dir, key });
}

pub fn ensure_cache_dir(allocator: std.mem.Allocator, base_dir: []const u8, key: []const u8) ![]u8 {
    const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, key });
    std.fs.cwd().makePath(dir) catch {};
    return dir;
}

pub const UpdateResult = struct {
    stable_path: []u8,
    best_candidate: usize,
    best_time_us: f64,
};

pub fn update_cache_from_work_dir(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    work_dir: []const u8,
    key: []const u8,
    target_kind: TargetKind,
) !UpdateResult {
    const best = try best_candidate_from_records(allocator, work_dir);

    const candidate_path = try std.fmt.allocPrint(allocator, "{s}/candidate_{d}.so", .{
        work_dir, best.idx,
    });
    defer allocator.free(candidate_path);

    const stable_path = try stable_artifact_path(allocator, work_dir, key);

    std.fs.cwd().copyFile(candidate_path, std.fs.cwd(), stable_path, .{}) catch |err| {
        allocator.free(stable_path);
        return err;
    };

    const entry = CacheEntry{
        .key = key,
        .target_kind = target_kind,
        .artifact_path = stable_path,
        .best_time_us = best.time_secs * 1e6,
    };
    try cache_update(allocator, base_dir, entry);

    return .{
        .stable_path = stable_path,
        .best_candidate = best.idx,
        .best_time_us = best.time_secs * 1e6,
    };
}

pub fn load_cached(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    key: []const u8,
    target_kind: TargetKind,
) !?TunedModule {
    const cached = try cache_lookup(allocator, base_dir, key, target_kind);
    if (cached == null) return null;
    defer {
        allocator.free(cached.?.key);
        allocator.free(cached.?.artifact_path);
    }

    std.fs.cwd().access(cached.?.artifact_path, .{}) catch return null;
    const path_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{cached.?.artifact_path}, 0);
    defer allocator.free(path_z);
    var module = try RuntimeModule.load_from_file(allocator, path_z);
    errdefer module.deinit();
    const main_func = try module.get_function(allocator, "main", true);

    return .{
        .module = module,
        .main_func = main_func,
        .best_candidate = 0,
        .best_time_us = cached.?.best_time_us,
    };
}

pub fn matmul_cache_key(
    allocator: std.mem.Allocator,
    target_kind: TargetKind,
    m: usize,
    n: usize,
    k: usize,
) ![]u8 {
    const target_suffix = switch (target_kind) {
        .cpu => "cpu",
        .cuda => "cuda",
    };
    const key_str = try std.fmt.allocPrint(allocator, "matmul:f32:{s}:{d}:{d}:{d}", .{
        target_suffix, m, n, k,
    });
    defer allocator.free(key_str);

    const hash = std.hash.Wyhash.hash(0, key_str);
    return std.fmt.allocPrint(allocator, "{x}", .{hash});
}

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
/// Scans backwards from the end for the run_secs field, which sits between the
/// decisions array and the target object: `...decisions], [<run_secs>], {target...`.
/// Handles both float (`1.85e-05`) and integer (`10000000000`) representations.
fn parse_run_secs(line: []const u8) ?f64 {
    // Scan backwards from end to find ],{ (boundary between run_secs array and target object)
    var i: usize = line.len;
    while (i > 4) {
        i -= 1;
        if (line[i] == '{' and line[i - 1] == ',' and line[i - 2] == ']') {
            // Found ],{ — now find the opening [ of the run_secs array
            var j = i - 3;
            while (j > 0 and line[j] != '[') j -= 1;
            if (j > 0 and line[j] == '[') {
                const num_str = line[j + 1 .. i - 2];
                return std.fmt.parseFloat(f64, num_str) catch null;
            }
        }
    }
    return null;
}

/// Load a specific candidate module by index.
fn load_candidate(allocator: std.mem.Allocator, work_dir: []const u8, candidate_idx: usize) !TunedModule {
    const so_path = try std.fmt.allocPrintSentinel(allocator, "{s}/candidate_{d}.so", .{ work_dir, candidate_idx }, 0);
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

fn read_cache_index(allocator: std.mem.Allocator, base_dir: []const u8) !CacheIndex {
    const path = try std.fmt.allocPrint(allocator, "{s}/index.json", .{base_dir});
    defer allocator.free(path);

    const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return .{ .entries = try allocator.alloc(CacheEntry, 0) },
        else => return err,
    };
    defer file.close();

    const file_size = try file.getEndPos();
    if (file_size == 0) return .{ .entries = try allocator.alloc(CacheEntry, 0) };

    const contents = try allocator.alloc(u8, file_size);
    defer allocator.free(contents);
    const bytes_read = try file.readAll(contents);

    const parsed = try std.json.parseFromSlice(CacheIndex, allocator, contents[0..bytes_read], .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    var entries = try allocator.alloc(CacheEntry, parsed.value.entries.len);
    for (parsed.value.entries, 0..) |entry, i| {
        entries[i] = .{
            .key = try allocator.dupe(u8, entry.key),
            .target_kind = entry.target_kind,
            .artifact_path = try allocator.dupe(u8, entry.artifact_path),
            .best_time_us = entry.best_time_us,
        };
    }
    return .{ .version = parsed.value.version, .entries = entries };
}

fn write_cache_index(allocator: std.mem.Allocator, base_dir: []const u8, index: CacheIndex) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/index.json", .{base_dir});
    defer allocator.free(path);

    const bytes = try std.json.Stringify.valueAlloc(allocator, index, .{});
    defer allocator.free(bytes);

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(bytes);
}

fn deinit_cache_index(allocator: std.mem.Allocator, index: *CacheIndex) void {
    for (index.entries) |entry| {
        allocator.free(entry.key);
        allocator.free(entry.artifact_path);
    }
    allocator.free(index.entries);
    index.* = undefined;
}
