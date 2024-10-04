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
load("@zg_py_test_deps//:requirements.bzl", "requirement")

py_binary(
    name = "generate_test_data",
    srcs = ["scripts/mnist_data.py"],
    main = "scripts/mnist_data.py",
    deps = [
        requirement("torch"),
        requirement("torchvision"),
    ],
)

genrule(
    name = "mnist_data",
    outs = [
        "test_data/zigrad_test_mnist_train_full.csv",
        "test_data/zigrad_test_mnist_test_full.csv",
        "test_data/zigrad_test_mnist_train_small.csv",
        "test_data/zigrad_test_mnist_test_small.csv",
    ],
    cmd = "ZG_TEST_DATA_DIR=$(@D)/test_data $(location :generate_test_data)",
    tools = [":generate_test_data"],
)

py_binary(
    name = "mnist_torch",
    srcs = ["src/nn/tests/test_mnist.py"],
    main = "src/nn/tests/test_mnist.py",
    data = [":mnist_data"],
    env = {"ZG_TEST_DATA_DIR": "./test_data"},
    deps = [
      requirement("torch"),
      requirement("numpy"),
    ],
)
