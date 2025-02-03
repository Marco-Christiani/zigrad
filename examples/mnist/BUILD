# After build, can run like so:
# ZG_DATA_DIR=bazel-bin/data/ bazel-bin/simple
# ZG_DATA_DIR=bazel-bin/data/ bazel-bin/timing
genrule(
    name = "build",
    srcs = glob(["src/**/*.zig", "build.zig", "build.zig.zon"]),
    outs = ["main"],
    cmd = """
    ZG_DATA_DIR=$$(dirname $$(echo $(locations :mnist_data) | cut -d' ' -f1)) \
    ZIG_LOCAL_CACHE_DIR=$$(pwd)/zig-cache \
    ZIG_GLOBAL_CACHE_DIR=$$(pwd)/zig-cache \
    zig build -Doptimize=ReleaseFast && cp zig-out/bin/* $(@D)/\
    """,
    tools = [":mnist_data"],
)


load("@rules_python//python:defs.bzl", "py_runtime", "py_binary")

py_binary(
    name = "generate_test_data",
    srcs = ["mnist_data.py"],
    main = "mnist_data.py",
)

genrule(
    name = "mnist_data",
    outs = [
        "data/mnist_train_full.csv",
        "data/mnist_test_full.csv",
        "data/mnist_train_small.csv",
        "data/mnist_test_small.csv",
    ],
    cmd = "ZG_DATA_DIR=$(@D)/data $(location :generate_test_data)",
    tools = [":generate_test_data"],
)
