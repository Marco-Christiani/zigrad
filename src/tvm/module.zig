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
const Cache = @import("../cache.zig").Cache;

const log = std.log.scoped(.@"zg/tvm_loader");

/// Options for loading a tuned module.
pub const LoadOpts = struct {
    /// Cache pointing to the directory containing tuning artifacts.
    work_cache: Cache,
    /// Target to load tuned modules for.
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

const current_schema_version: u32 = 1;

const CacheIndex = struct {
    schema_version: u32 = current_schema_version,
    file_version: u32 = 0,
    entries: []CacheEntry = &.{},
};

/// Fixed-size buffer for cache keys (wyhash u64, up to 16 hex chars).
pub const CacheKey = struct {
    buf: [16]u8 = undefined,
    len: u8 = 0,

    pub fn slice(self: *const CacheKey) []const u8 {
        return self.buf[0..self.len];
    }
};

/// Compute a deterministic cache key for a matmul shape + target.
/// **Assumes f32.**
pub fn matmul_cache_key(target_kind: TargetKind, m: i64, n: i64, k: i64) CacheKey {
    const target_suffix: []const u8 = @tagName(target_kind);
    var input_buf: [128]u8 = undefined;
    const key_str = std.fmt.bufPrint(&input_buf, "matmul:f32:{s}:{d}:{d}:{d}", .{
        target_suffix, m, n, k,
    }) catch unreachable;
    const hash = std.hash.Wyhash.hash(0, key_str);
    var result: CacheKey = .{};
    const formatted = std.fmt.bufPrint(&result.buf, "{x}", .{hash}) catch unreachable;
    result.len = @intCast(formatted.len);
    return result;
}

pub fn cache_lookup(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_cache: Cache,
    key: []const u8,
    target_kind: TargetKind,
) !?CacheEntry {
    var index = try read_cache_index(io, allocator, base_cache);
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
    io: std.Io,
    allocator: std.mem.Allocator,
    base_cache: Cache,
    entry: CacheEntry,
) !void {
    var index = try read_cache_index(io, allocator, base_cache);
    defer deinit_cache_index(allocator, &index);

    for (index.entries) |*existing| {
        if (existing.target_kind == entry.target_kind and std.mem.eql(u8, existing.key, entry.key)) {
            allocator.free(existing.key);
            allocator.free(existing.artifact_path);
            existing.key = try allocator.dupe(u8, entry.key);
            existing.artifact_path = try allocator.dupe(u8, entry.artifact_path);
            existing.best_time_us = entry.best_time_us;
            return write_cache_index(io, allocator, base_cache, &index);
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
    return write_cache_index(io, allocator, base_cache, &index);
}

pub fn best_candidate_from_records(io: std.Io, allocator: std.mem.Allocator, work_cache: Cache) !RankedCandidate {
    const candidates = try find_ranked_candidates(io, allocator, work_cache);
    defer allocator.free(candidates);
    return candidates[0];
}

pub fn stable_artifact_path(work_cache: Cache, key: []const u8) !Cache {
    var name_buf: [256]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "kernel_{s}.so", .{key}) catch return error.NameTooLong;
    return work_cache.join(name);
}

pub const UpdateResult = struct {
    stable_path: Cache,
    best_candidate: usize,
    best_time_us: f64,
};

pub fn update_cache_from_work_dir(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_cache: Cache,
    work_cache: Cache,
    key: []const u8,
    target_kind: TargetKind,
) !UpdateResult {
    const best = try best_candidate_from_records(io, allocator, work_cache);

    var candidate_name_buf: [64]u8 = undefined;
    const candidate_name = std.fmt.bufPrint(&candidate_name_buf, "candidate_{d}.so", .{best.idx}) catch unreachable;
    const candidate = try work_cache.join(candidate_name);

    const stable = try stable_artifact_path(work_cache, key);

    const cwd = std.Io.Dir.cwd();
    cwd.copyFile(candidate.path(), cwd, stable.path(), io, .{ .replace = true }) catch |err| {
        return err;
    };

    const entry = CacheEntry{
        .key = key,
        .target_kind = target_kind,
        .artifact_path = stable.path(),
        .best_time_us = best.time_secs * 1e6,
    };
    try cache_update(io, allocator, base_cache, entry);

    return .{
        .stable_path = stable,
        .best_candidate = best.idx,
        .best_time_us = best.time_secs * 1e6,
    };
}

pub fn load_cached(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_cache: Cache,
    key: []const u8,
    target_kind: TargetKind,
) !?TunedModule {
    const cached = try cache_lookup(io, allocator, base_cache, key, target_kind);
    if (cached == null) return null;
    defer {
        allocator.free(cached.?.key);
        allocator.free(cached.?.artifact_path);
    }

    std.Io.Dir.cwd().access(io, cached.?.artifact_path, .{}) catch return null;
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

/// Load the best tuned module from a previous tuning run.
///
/// Parses tuning_record.json to find the fastest candidate, then loads
/// the corresponding .so file and returns a handle to the main function.
pub fn load(io: std.Io, allocator: std.mem.Allocator, opts: LoadOpts) !TunedModule {
    const candidates = try find_ranked_candidates(io, allocator, opts.work_cache);
    defer allocator.free(candidates);

    const best = candidates[0];
    var tuned = try load_candidate(allocator, opts.work_cache, best.idx);
    tuned.best_time_us = best.time_secs * 1e6;

    log.info("loaded tuned module (candidate {d}, {d:.2} us)", .{ best.idx, tuned.best_time_us });
    return tuned;
}

// ============================================================================
// Private helpers
// ============================================================================

pub const RankedCandidate = struct {
    idx: usize,
    time_secs: f64,
};

/// Parse tuning_record.json and return candidates ranked by speed (fastest first).
///
/// Only includes candidates whose .so file exists on disk.
fn find_ranked_candidates(io: std.Io, allocator: std.mem.Allocator, work_cache: Cache) ![]RankedCandidate {
    const record_file = try work_cache.join("tuning_record.json");
    const record_path = record_file.path();

    const contents = std.Io.Dir.cwd().readFileAlloc(io, record_path, allocator, .unlimited) catch |err| {
        log.err("failed to read tuning records at {s}: {s}", .{ record_path, @errorName(err) });
        return error.NoTuningRecords;
    };
    defer allocator.free(contents);

    if (contents.len == 0) {
        log.err("empty tuning records file: {s}", .{record_path});
        return error.NoTuningRecords;
    }
    const bytes_read = contents.len;

    var candidates = std.ArrayList(RankedCandidate).empty;
    defer candidates.deinit(allocator);
    var line_num: usize = 0;

    var lines = std.mem.splitScalar(u8, contents[0..bytes_read], '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const run_secs = parse_run_secs(line);

        if (run_secs) |t| {
            var name_buf: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "candidate_{d}.so", .{line_num}) catch unreachable;
            const candidate = work_cache.join(name) catch {
                line_num += 1;
                continue;
            };

            std.Io.Dir.cwd().access(io, candidate.path(), .{}) catch {
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
    var i: usize = line.len;
    while (i > 4) {
        i -= 1;
        if (line[i] == '{' and line[i - 1] == ',' and line[i - 2] == ']') {
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
fn load_candidate(allocator: std.mem.Allocator, work_cache: Cache, candidate_idx: usize) !TunedModule {
    var name_buf: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "candidate_{d}.so", .{candidate_idx}) catch unreachable;
    var so = try work_cache.join(name);
    const so_path = so.pathZ();

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

fn read_cache_index(io: std.Io, allocator: std.mem.Allocator, base_cache: Cache) !CacheIndex {
    const idx_file = try base_cache.join("index.json");
    const path = idx_file.path();

    const contents = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return .{ .entries = try allocator.alloc(CacheEntry, 0) },
        else => return err,
    };
    defer allocator.free(contents);

    if (contents.len == 0) return .{ .entries = try allocator.alloc(CacheEntry, 0) };

    const parsed = try std.json.parseFromSlice(CacheIndex, allocator, contents, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    if (parsed.value.schema_version != current_schema_version) {
        log.warn("index.json schema_version {d}, expected {d}", .{
            parsed.value.schema_version, current_schema_version,
        });
    }

    var entries = try allocator.alloc(CacheEntry, parsed.value.entries.len);
    for (parsed.value.entries, 0..) |entry, i| {
        entries[i] = .{
            .key = try allocator.dupe(u8, entry.key),
            .target_kind = entry.target_kind,
            .artifact_path = try allocator.dupe(u8, entry.artifact_path),
            .best_time_us = entry.best_time_us,
        };
    }
    return .{
        .schema_version = parsed.value.schema_version,
        .file_version = parsed.value.file_version,
        .entries = entries,
    };
}

fn write_cache_index(io: std.Io, allocator: std.mem.Allocator, base_cache: Cache, index: *CacheIndex) !void {
    const idx_file = try base_cache.join("index.json");
    const path = idx_file.path();

    // Serialize current state to compare against file on disk.
    const bytes = try std.json.Stringify.valueAlloc(allocator, index.*, .{});
    defer allocator.free(bytes);

    const old_bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024)) catch null;
    if (old_bytes) |old| {
        defer allocator.free(old);
        if (std.mem.eql(u8, old, bytes)) return;
    }

    // Content changed: bump file_version, re-serialize, write
    index.file_version += 1;
    const final_bytes = try std.json.Stringify.valueAlloc(allocator, index.*, .{});
    defer allocator.free(final_bytes);

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, final_bytes);
}

fn deinit_cache_index(allocator: std.mem.Allocator, index: *CacheIndex) void {
    for (index.entries) |entry| {
        allocator.free(entry.key);
        allocator.free(entry.artifact_path);
    }
    allocator.free(index.entries);
    index.* = undefined;
}
