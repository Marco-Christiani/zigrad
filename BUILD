# load(
#   "@rules_zig//zig:defs.bzl",
#   "zig_binary",
#   "zig_library",
#   "zig_module",
# )
#
# # Module target
# zig_module(
#     name = "zigrad",
#     main = "src/root.zig",
#     srcs = glob(["src/**/*.zig"]),
#     extra_srcs = ["src/scalar/data.csv"],
# )
#
# # Library target
# zig_library(
#     name = "zigrad-lib",
#     main = "src/root.zig",
#     deps = [
#         ":zigrad",
#     ],
# )
#
# # Binary target
# zig_binary(
#     name = "zigrad.bin",
#     main = "src/main.zig",
#     deps = [
#         ":zigrad",
#     ],
# )
#

# Zig --------------------------------------------------------------------------
genrule(
    name = "build_zigrad",
    srcs = glob(["src/**/*.zig", "src/scalar/data.csv", "build.zig"]),
    outs = ["zig-out/lib/libzigrad.a"],
    cmd = "ZIG_LOCAL_CACHE_DIR=$$(pwd)/zig-cache ZIG_GLOBAL_CACHE_DIR=$$(pwd)/zig-cache zig build -Doptimize=ReleaseFast && cp zig-out/lib/libzigrad.a $@",
)

load("@rules_python//python:defs.bzl", "py_runtime", "py_binary")
load("@rules_python//python:pip.bzl", "pip_install")

py_runtime(
    name = "python3_8_runtime",
    interpreter_path = "/path/to/python3.8/bin/python3",
    files = [],
)

pip_install(
    name = "pip_deps",
    packages = {
        "torch": "1.9.0",
        "numpy": "1.19.2",
    },
)

py_binary(
    name = "generate_test_data",
    srcs = ["tests/generate_test_data.py"],
    main = "tests/generate_test_data.py",
    runtime = ":python3_8_runtime",
    deps = [":pip_deps"],
)
