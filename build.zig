const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Canonical external SDK root containing:
    //   - include/  (mlir-c, stablehlo, xla/pjrt/c)
    //   - lib/      (libMLIR-C.so, libStablehloCAPI.so, LLVM/MLIR deps)
    //   - runtime/  (xla/pjrt/c plugins + bundled CUDA user-space libs)
    // TODO: should this support ZG_EXTERNAL_SDK? did we remove that?
    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad external SDK root (include/, lib/, runtime/)") orelse "./result";

    // Dev convenience: override runtime bundle root directory.
    // If runtime link installation is enabled, we symlink `zig-out/runtime` to this path.
    const runtime_root_opt = b.option([]const u8, "runtime", "Override runtime bundle root (dev convenience)");
    const install_runtime_link = b.option(bool, "install-runtime-link", "Create zig-out/runtime symlink (dev convenience)") orelse false;
    const autodoc_web_dir_opt = b.option([]const u8, "autodoc-web-dir", "Nuxt public/api target for `docs-web` sync") orelse "website/nuxt-content/public/api";

    const sdk_include = b.fmt("{s}/include", .{sdk_root});
    const sdk_lib = b.fmt("{s}/lib", .{sdk_root});
    const sdk_runtime = b.fmt("{s}/runtime", .{sdk_root});

    // Feature detection: sdk_has_* probes for headers in the SDK.
    // Feature flags: -Dx=true forces on (skips detection; @cImport fails if headers missing),
    //                -Dx=false forces off, omitted = auto-detect via sdk_has_*.
    const enable_mlir = b.option(bool, "mlir", "Force MLIR/StableHLO lowering on or off");
    const enable_tvm = b.option(bool, "tvm", "Force TVM kernel provider on or off");
    const enable_mirage = b.option(bool, "mirage", "Force Mirage kernel provider on or off");
    const enable_iree = b.option(bool, "iree-backend", "Force IREE backend on or off");

    // When enabled, the op registry emits @compileLog's of the op interface coverage matrix.
    const emit_op_coverage = b.option(bool, "emit-op-coverage", "Emit op interface coverage at comptime (fails the build)") orelse false;

    const has_mlir = sdk_has_mlir(b, sdk_root);
    const has_tvm = sdk_has_tvm(b, sdk_root);
    const has_mirage = sdk_has_mirage(b, sdk_root);

    const use_mlir = enable_mlir orelse has_mlir;
    const use_tvm = enable_tvm orelse has_tvm;
    const use_mirage = enable_mirage orelse has_mirage;
    const use_iree = enable_iree orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "has_mlir", use_mlir);
    build_options.addOption(bool, "has_tvm", use_tvm);
    build_options.addOption(bool, "has_mirage", use_mirage);
    build_options.addOption(bool, "has_iree", use_iree);
    build_options.addOption(bool, "emit_op_coverage", emit_op_coverage);

    const safetensors_zg_dep = b.dependency("safetensors_zg", .{});
    const cova_dep = b.dependency("cova", .{});
    const protobuf_dep = b.dependency("protobuf", .{});
    const protobuf_mod = protobuf_dep.module("protobuf");
    const xla_pb_mod = b.createModule(.{
        .root_source_file = b.path("src/c/xla/proto/xla.pb.zig"),
        .imports = &.{
            .{ .name = "protobuf", .module = protobuf_mod },
        },
    });
    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zigrad_mod.addOptions("build_options", build_options);
    zigrad_mod.addImport("protobuf", protobuf_mod);
    zigrad_mod.addImport("xla_pb", xla_pb_mod);
    zigrad_mod.addImport("safetensors_zg", safetensors_zg_dep.module("safetensors_zg"));
    zigrad_mod.addIncludePath(b.path("src"));
    zigrad_mod.addIncludePath(.{ .cwd_relative = sdk_include });

    // Add CUDA include path if available (needed for nvrtc.h).
    // TODO: should be exposed as a build option so consumers can set it in their build,
    //  supporting the env var fallback is debatably acceptable, but we should be
    //  consistent about policy wrt the sdk root.
    const cuda_include_opt: ?[]const u8 = if (b.graph.environ_map.get("CUDA_HOME")) |cuda_home|
        b.fmt("{s}/include", .{cuda_home})
    else
        null;
    if (cuda_include_opt) |cuda_include|
        zigrad_mod.addIncludePath(.{ .cwd_relative = cuda_include });

    // C API translation (replaces former @cImport sites in src/c/*).
    //  Each TranslateC step produces a private module that we import into
    //  `zigrad_mod` under a stable name; the matching `src/c/*.zig` wrapper
    //  swaps `@cImport` for `@import("c-foo")`.
    var c_iree_mod: ?*std.Build.Module = null;
    {
        const c_pjrt = b.addTranslateC(.{
            .root_source_file = b.path("src/c/pjrt/headers.h"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        c_pjrt.addIncludePath(.{ .cwd_relative = sdk_include });
        zigrad_mod.addImport("c-pjrt", c_pjrt.createModule());

        if (use_mlir) {
            const c_mlir = b.addTranslateC(.{
                .root_source_file = b.path("src/c/mlir/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_mlir.addIncludePath(.{ .cwd_relative = sdk_include });
            zigrad_mod.addImport("c-mlir", c_mlir.createModule());
        }

        if (use_tvm) {
            const c_tvm = b.addTranslateC(.{
                .root_source_file = b.path("src/c/tvm/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_tvm.addIncludePath(.{ .cwd_relative = sdk_include });
            zigrad_mod.addImport("c-tvm", c_tvm.createModule());
        }

        if (use_mirage) {
            const c_mirage = b.addTranslateC(.{
                .root_source_file = b.path("src/c/mirage/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_mirage.addIncludePath(.{ .cwd_relative = sdk_include });
            zigrad_mod.addImport("c-mirage", c_mirage.createModule());
        }

        if (cuda_include_opt) |cuda_include| {
            const c_nvrtc = b.addTranslateC(.{
                .root_source_file = b.path("src/c/nvrtc_headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_nvrtc.addIncludePath(.{ .cwd_relative = cuda_include });
            zigrad_mod.addImport("c-nvrtc", c_nvrtc.createModule());
        }

        if (use_iree) {
            const c_iree = b.addTranslateC(.{
                .root_source_file = b.path("src/c/iree/iree_zig.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_iree.addIncludePath(.{ .cwd_relative = sdk_include });
            const mod = c_iree.createModule();
            zigrad_mod.addImport("c-iree", mod);
            c_iree_mod = mod;
        }
    }

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
                .{ .name = "safetensors_zg", .module = safetensors_zg_dep.module("safetensors_zg") },
                .{ .name = "cova", .module = cova_dep.module("cova") },
            },
        }),
    });
    exe.root_module.addIncludePath(b.path("src"));
    exe.root_module.addIncludePath(.{ .cwd_relative = sdk_include });
    if (use_mlir) link_mlir_stablehlo_capi(exe, sdk_lib);
    if (use_iree) link_iree(zigrad_mod, sdk_lib);
    add_runtime_bundle(b, exe, runtime_root_opt orelse sdk_runtime, install_runtime_link);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the v0 demo executable").dependOn(&run_cmd.step);

    const lib_tests = b.addTest(.{ .root_module = zigrad_mod });
    if (use_mlir) link_mlir_stablehlo_capi(lib_tests, sdk_lib);
    add_runtime_bundle(b, lib_tests, runtime_root_opt orelse sdk_runtime, install_runtime_link);

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);

    // Emit library autodocs to zig-out/autodoc.
    const docs_obj = b.addObject(.{
        .name = "zigrad",
        .root_module = zigrad_mod,
    });
    const install_autodoc = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "autodoc",
    });
    const docs_step = b.step("docs", "Emit Zig autodocs to zig-out/autodoc");
    docs_step.dependOn(&install_autodoc.step);

    // Emit docs and mirror them into the web app's public/api directory.
    const autodoc_web_dir = resolve_absolute_path(b, autodoc_web_dir_opt);
    const sync_autodoc_to_web = b.addSystemCommand(&[_][]const u8{
        "bash",
        "-lc",
        std.fmt.allocPrint(b.allocator,
            \\set -euo pipefail
            \\src="{s}/autodoc"
            \\dst="{s}"
            \\mkdir -p "$dst"
            \\cp -fR "$src/." "$dst/"
            \\chmod -R u+w "$dst"
        , .{ b.install_prefix, autodoc_web_dir }) catch @panic("OOM"),
    });
    sync_autodoc_to_web.step.dependOn(&install_autodoc.step);
    const docs_web_step = b.step("docs-web", "Emit Zig autodocs and sync website/nuxt-content/public/api");
    docs_web_step.dependOn(&sync_autodoc_to_web.step);

    // CLI completion + manpage gen. Runs the built binary at build time,
    //  which fails inside hermetic sandboxes (e.g. nix) where the default
    //  ELF interpreter path is unavailable. Producing completions is
    //  packaging metadata and not part of the runtime build, so it is gated
    //  behind a flag that nix turns off.
    const gen_cli_meta = b.option(bool, "gen-cli-meta", "Generate shell completions and manpages (runs the built exe)") orelse true;
    if (gen_cli_meta) {
        const gen_completions = add_cli_gen_step(b, cova_dep, exe);
        gen_completions.step.dependOn(&exe.step);
        b.getInstallStep().dependOn(&gen_completions.step);
    }

    // HLO protobuf decode tool (standalone, no SDK dependencies).
    // Reuses protobuf_mod, xla_pb_mod hoisted above.
    {
        const hlo_decode_mod = b.createModule(.{
            .root_source_file = b.path("src/c/xla/hlo_decode.zig"),
            .imports = &.{
                .{ .name = "protobuf", .module = protobuf_mod },
                .{ .name = "xla_pb", .module = xla_pb_mod },
            },
        });

        const decode_hlo = b.addExecutable(.{
            .name = "decode_hlo",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/decode_hlo.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "protobuf", .module = protobuf_mod },
                    .{ .name = "xla_pb", .module = xla_pb_mod },
                    .{ .name = "hlo_decode", .module = hlo_decode_mod },
                },
            }),
        });
        const install = b.addInstallArtifact(decode_hlo, .{});
        b.getInstallStep().dependOn(&install.step);
        b.step("decode-hlo", "Build HLO protobuf decoder tool").dependOn(&install.step);
    }

    // Minimal IREE VMFB runner (no zigrad, no MLIR/PJRT).
    if (use_iree) {
        const iree_runner_mod = b.createModule(.{
            .root_source_file = b.path("src/iree_runner.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        if (c_iree_mod) |m| iree_runner_mod.addImport("c-iree", m);
        const iree_runner = b.addExecutable(.{
            .name = "iree-runner",
            .root_module = iree_runner_mod,
        });
        iree_runner.root_module.addIncludePath(b.path("src"));
        iree_runner.root_module.addIncludePath(.{ .cwd_relative = sdk_include });
        link_iree(iree_runner.root_module, sdk_lib);
        b.installArtifact(iree_runner);
        b.step("iree-runner", "Build minimal IREE VMFB runner").dependOn(&iree_runner.step);
    }
}

/// Add CLI completion/manpage gen step
fn add_cli_gen_step(
    b: *std.Build,
    cova_dep: *std.Build.Dependency,
    exe: *std.Build.Step.Compile,
) *std.Build.Step.Run {
    // Cova's generator.zig requires three option modules (`md_config_opts`,
    // `tab_complete_config`, `help_docs_config`, `arg_template_config`). When
    // `provided: true`, `optsToConf` reads EVERY field from the config struct out
    // of the options module, so it must have all fields. When `provided: false`,
    // it returns null immediately and no fields are read.
    const cova_gen_exe = b.addExecutable(.{
        .name = "zigrad_completion_generator",
        .root_module = b.createModule(.{
            .root_source_file = cova_dep.path("src/generator.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseSafe,
        }),
    });
    cova_gen_exe.root_module.addImport("cova", cova_dep.module("cova"));
    cova_gen_exe.root_module.addImport("program", exe.root_module);

    // Shared metadata
    // `kinds`: which doc types to generate (ordinals, see `MetaDocConfig.MetaDocKind`)
    //          all=0 manpages=1 markdown=2 bash=3 zsh=4 ps1=5 fish=6 json=7 kdl=8
    // The remaining fields override null values in per-config modules.
    const md_conf_opts = b.addOptions();
    md_conf_opts.addOption([]const usize, "kinds", &[_]usize{ 1, 3, 4, 5, 6 }); // manpages, bash, zsh, ps1, fish
    md_conf_opts.addOption([]const u8, "cmd_type_name", "CommandT"); // pub decl name in src/main.zig
    md_conf_opts.addOption([]const u8, "setup_cmd_name", "setup_cmd"); // pub decl name in src/main.zig
    md_conf_opts.addOption(?[]const u8, "name", null); // null -> uses CommandT.name ("zigrad")
    md_conf_opts.addOption(?[]const u8, "description", null); // null -> uses CommandT.description
    md_conf_opts.addOption(?[]const u8, "version", "dev");
    md_conf_opts.addOption(?[]const u8, "ver_date", null);
    md_conf_opts.addOption(?[]const u8, "author", null);
    md_conf_opts.addOption(?[]const u8, "copyright", null);
    md_conf_opts.addOption(usize, "log_level", @intFromEnum(std.log.Level.warn)); // suppress generator info logs
    cova_gen_exe.root_module.addOptions("md_config_opts", md_conf_opts);

    // TabCompletionConfig fields
    const tab_conf = b.addOptions();
    tab_conf.addOption(bool, "provided", true);
    tab_conf.addOption([]const u8, "local_filepath", b.fmt("{s}/completions/", .{b.install_prefix}));
    tab_conf.addOption(?[]const u8, "script_header", null); // null -> per-shell default shebang
    tab_conf.addOption(?[]const u8, "name", null); // null -> falls back to md_config_opts.name
    tab_conf.addOption(bool, "include_cmds", true); // include subcommands in completions
    tab_conf.addOption(bool, "include_opts", true); // include --options in completions
    tab_conf.addOption(bool, "include_usage_help", true); // include help/usage pseudo-commands
    tab_conf.addOption(bool, "add_cova_lib_msg", true); // "generated by Cova" header comment
    tab_conf.addOption(bool, "add_install_instructions", true); // shell-specific install instructions
    cova_gen_exe.root_module.addOptions("tab_complete_config", tab_conf);

    // HelpDocsConfig fields (manpages enabled)
    // When provided=true, optsToConf reads EVERY HelpDocsConfig field from
    // this module. Shared metadata fields (name, version, etc.) are set null
    // here and overridden from md_config_opts by optsToConf.
    const help_conf = b.addOptions();
    help_conf.addOption(bool, "provided", true);
    help_conf.addOption([]const u8, "local_filepath", b.fmt("{s}/share/man/", .{b.install_prefix}));
    // Recursion
    help_conf.addOption(bool, "recursive_gen", true);
    help_conf.addOption(u8, "recursive_max_depth", 3);
    help_conf.addOption([]const []const u8, "recursive_blocklist", &.{ "usage", "help" });
    // Shared metadata -- null -> inherited from md_config_opts
    help_conf.addOption(?[]const u8, "version", null);
    help_conf.addOption(?[]const u8, "ver_date", null);
    help_conf.addOption(?[]const u8, "name", null);
    help_conf.addOption(?[]const u8, "description", null);
    help_conf.addOption(?[]const u8, "author", null);
    help_conf.addOption(?[]const u8, "copyright", null);
    help_conf.addOption(?[]const u8, "examples", null);
    // Manpage metadata
    help_conf.addOption(u8, "section", '1');
    help_conf.addOption(?[]const u8, "man_name", "zigrad");
    help_conf.addOption(?[]const u8, "synopsis", null);
    // Manpage format strings (defaults match HelpDocsConfig)
    help_conf.addOption([]const u8, "mp_subcmds_fmt", ".B {s}:\n{s}\n\n");
    help_conf.addOption([]const u8, "mp_opts_fmt", ".B {s}:\n[{u}{?u},{s}{?s} \"{s} ({s})\"]:\n  {s}\n\n");
    help_conf.addOption([]const u8, "mp_vals_fmt", ".B {s}:\n({s}): {s}\n\n");
    help_conf.addOption([]const u8, "mp_examples_fmt", ".B {s}\n\n");
    // Markdown format strings (defaults match HelpDocsConfig)
    help_conf.addOption([]const u8, "md_subcmds_fmt", "- [__{s}__]({s}): {s}\n");
    help_conf.addOption([]const u8, "md_opts_fmt", "- __{s}__:\n    - `{u}{u}{s}{s}{s}{s} <{s} ({s})>`\n    - {s}\n");
    help_conf.addOption([]const u8, "md_opt_names_sep_fmt", ", ");
    help_conf.addOption([]const u8, "md_vals_fmt", "- __{s}__ ({s})\n    - {s}\n");
    help_conf.addOption([]const u8, "md_examples_fmt", "- `{s}`\n");
    cova_gen_exe.root_module.addOptions("help_docs_config", help_conf);

    // ArgTemplateConfig stub (unused)
    // `generator.zig` unconditionally @imports this module so we must provide it.
    // `provided: false` -> optsToConf returns null, no fields are read.
    // To enable: set provided=true, add ALL ArgTemplateConfig fields, and add
    // json=7 / kdl=8 to md_conf_opts.kinds.
    const arg_conf = b.addOptions();
    arg_conf.addOption(bool, "provided", false);
    arg_conf.addOption([]const u8, "local_filepath", "arg_templates");
    cova_gen_exe.root_module.addOptions("arg_template_config", arg_conf);

    return b.addRunArtifact(cova_gen_exe);
}

fn link_mlir_stablehlo_capi(exe: *std.Build.Step.Compile, sdk_lib: []const u8) void {
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });

    // MLIR + StableHLO were built with libstdc++ ABI.
    exe.root_module.linkSystemLibrary("stdc++", .{});

    // Shared MLIR C boundary (the SDK provides libMLIR-C.so).
    exe.root_module.linkSystemLibrary("MLIR-C", .{});

    // StableHLO C API boundary (libStablehloCAPI.so provides stablehlo dialect handle symbols).
    exe.root_module.linkSystemLibrary("StablehloCAPI", .{});
}

/// Link the IREE runtime static archives and compile the IREE C shim.
///
/// Added to `zigrad_mod` so the link artifacts propagate once through
/// module imports (avoids duplicate symbols when the completion generator
/// transitively imports `zigrad_mod`).
///
/// libIREECompiler.so is NOT linked at build time.  Compilation is handled
/// out-of-process via `iree-compile` (subprocess) or optionally via dlopen.
/// The runtime is linked statically from archives produced by nix/iree-runtime.nix.
///
/// The shim (`src/c/iree/shim.c`) wraps `static inline` functions and
/// macros from the IREE headers that `@cImport` cannot translate.
fn link_iree(mod: *std.Build.Module, sdk_lib: []const u8) void {
    const b = mod.owner;

    // Link IREE runtime static archive directly.
    mod.addObjectFile(.{ .cwd_relative = b.fmt("{s}/libiree_runtime_unified.a", .{sdk_lib}) });

    // flatcc archives (IREE's FlatBuffer dependency).
    for ([_][]const u8{ "libflatcc_parsing.a", "libflatcc_runtime.a" }) |name| {
        const path = b.fmt("{s}/{s}", .{ sdk_lib, name });
        if (std.Io.Dir.cwd().access(b.graph.io, path, .{})) |_| {
            mod.addObjectFile(.{ .cwd_relative = path });
        } else |_| {}
    }

    // Compile the C shim that wraps IREE static inline / macro helpers.
    mod.addCSourceFile(.{
        .file = b.path("src/c/iree/shim.c"),
        .flags = &.{
            "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl",
        },
    });
}

fn sdk_has_mlir(b: *std.Build, sdk_root: []const u8) bool {
    const header_path = b.pathJoin(&.{ sdk_root, "include", "mlir-c", "IR.h" });
    std.Io.Dir.cwd().access(b.graph.io, header_path, .{}) catch return false;
    return true;
}

fn sdk_has_tvm(b: *std.Build, sdk_root: []const u8) bool {
    const header_path = b.pathJoin(&.{ sdk_root, "include", "tvm", "ffi", "c_api.h" });
    std.Io.Dir.cwd().access(b.graph.io, header_path, .{}) catch return false;
    return true;
}

fn sdk_has_mirage(b: *std.Build, sdk_root: []const u8) bool {
    const header_path = b.pathJoin(&.{ sdk_root, "include", "mirage", "c", "types.h" });
    std.Io.Dir.cwd().access(b.graph.io, header_path, .{}) catch return false;
    return true;
}

fn resolve_absolute_path(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const build_root = b.build_root.path orelse ".";
    return std.fs.path.join(b.allocator, &.{ build_root, path }) catch @panic("path join failed");
}

fn add_runtime_bundle(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    runtime_root: []const u8,
    install_runtime_link: bool,
) void {
    exe.root_module.linkSystemLibrary("dl", .{});

    // RUNPATH entries for bare-soname dlopen resolution.
    //
    // Policy: include every SDK lib directory so that dlopen("libfoo.so")
    //  resolves automatically when the binary lives in the SDK tree.
    // Nonexistent directories (e.g. CUDA dirs in a CPU-only build) are
    //  silently skipped by the dynamic linker.
    //
    // NOTE: RUNPATH (DT_RUNPATH) does NOT propagate to transitive deps of
    //  dlopen'd libraries. Plugin transitive deps (e.g. PJRT plugin needing
    //  libcudart) must be handled by the plugin's own RUNPATH, this
    //  responsibility is delegated to the builder (e.g., our nix derivations)
    const rpaths = [_][]const u8{
        "$ORIGIN",
        "$ORIGIN/../lib", // build-time linked (MLIR-C, StablehloCAPI) + opt-in (TVM, Mirage)
        "$ORIGIN/../runtime/nvidia/nvrtc/lib",
        "$ORIGIN/../runtime/nvidia/cublas/lib",
        "$ORIGIN/../runtime/nvidia/cudart/lib",
        "$ORIGIN/../runtime/nvidia/cudnn/lib",
        "$ORIGIN/../runtime/nvidia/cufft/lib",
        "$ORIGIN/../runtime/nvidia/cupti/lib",
        "$ORIGIN/../runtime/nvidia/cusparse/lib",
        "$ORIGIN/../runtime/nvidia/nvjitlink/lib",
        "$ORIGIN/../runtime/nvidia/nccl/lib",
        "$ORIGIN/../runtime/nvidia/nvshmem/lib",
        "$ORIGIN/../runtime/sys/lib",
        "/run/opengl-driver/lib", // NixOS host GPU driver
    };
    inline for (rpaths) |p| exe.root_module.addRPathSpecial(p);

    if (!install_runtime_link) return;

    // Dev convenience: symlink `zig-out/runtime` -> runtime_root
    const runtime_root_abs = resolve_absolute_path(b, runtime_root);
    const link_step = b.addSystemCommand(&[_][]const u8{
        "bash",
        "-lc",
        std.fmt.allocPrint(b.allocator,
            \\set -euo pipefail
            \\prefix="{s}"
            \\mkdir -p "$prefix"
            \\ln -sfn "{s}" "$prefix/runtime"
        , .{ b.install_prefix, runtime_root_abs }) catch @panic("OOM"),
    });
    exe.step.dependOn(&link_step.step);
}
