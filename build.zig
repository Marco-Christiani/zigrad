const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda = b.option(bool, "cuda", "Target Cuda Backend") orelse false;

    generateBackend(cuda);

    const device_root = b.path(if (cuda) "src/cuda.zig" else "src/host.zig");
    const device = b.addModule("device", .{ .root_source_file = device_root });

    const zigrad = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .imports = &.{
            .{ .name = "device", .module = device },
        },
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("zigrad", zigrad);

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

pub fn generateBackend(cuda: bool) void {
    
    const backend: []const u8 = if (cuda) ".CUDA" else ".HOST";

    const format: []const u8 = 
        \\pub const Backend = enum{{ HOST, CUDA }};
        \\pub const backend: Backend = {s};
    ;

    var buffer: [128]u8 = undefined;

    const contents = std.fmt.bufPrint(buffer[0..], format, .{ backend }) catch unreachable;
    
    stringToFile("src/backend.zig", contents);
}

fn stringToFile(path: []const u8, string: []const u8) void {
    const end = std.mem.indexOfScalar(u8, string, 0) orelse string.len;

    var file = std.fs.cwd().createFile(path, .{}) catch @panic("Failed to create file.");
    defer file.close();

    var writer = file.writer();
    _ = writer.writeAll(string[0..end]) catch @panic("Failed to write file.");
}
