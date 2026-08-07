const std = @import("std");
const cli = @import("zigrad_cli_metadata");

const Renderer = *const fn (*std.Io.Writer) anyerror!void;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len < 2 or args.len > 3) {
        std.debug.print("usage: zigrad-cli-meta <install-prefix> [version]\n", .{});
        std.process.exit(1);
    }

    const prefix = args[1];
    const version = if (args.len == 3) args[2] else "dev";

    try render_file(
        io,
        allocator,
        prefix,
        "share/bash-completion/completions",
        "zigrad",
        cli.write_bash_completion,
    );
    try render_file(
        io,
        allocator,
        prefix,
        "share/zsh/site-functions",
        "_zigrad",
        cli.write_zsh_completion,
    );
    try render_file(
        io,
        allocator,
        prefix,
        "share/fish/vendor_completions.d",
        "zigrad.fish",
        cli.write_fish_completion,
    );
    try render_file(
        io,
        allocator,
        prefix,
        "share/powershell/Completions",
        "zigrad.ps1",
        cli.write_powershell_completion,
    );
    try render_manpage(io, allocator, prefix, version);
}

fn render_file(
    io: std.Io,
    allocator: std.mem.Allocator,
    prefix: []const u8,
    relative_dir: []const u8,
    name: []const u8,
    renderer: Renderer,
) !void {
    const dir_path = try std.fs.path.join(allocator, &.{ prefix, relative_dir });
    defer allocator.free(dir_path);
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

    const path = try std.fs.path.join(allocator, &.{ dir_path, name });
    defer allocator.free(path);

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try renderer(&output.writer);
    try write_bytes(io, path, output.written());
}

fn render_manpage(
    io: std.Io,
    allocator: std.mem.Allocator,
    prefix: []const u8,
    version: []const u8,
) !void {
    const dir_path = try std.fs.path.join(allocator, &.{ prefix, "share/man/man1" });
    defer allocator.free(dir_path);
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

    const path = try std.fs.path.join(allocator, &.{ dir_path, "zigrad.1" });
    defer allocator.free(path);

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();
    try cli.write_manpage(&output.writer, version);
    try write_bytes(io, path, output.written());
}

fn write_bytes(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}
