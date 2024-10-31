genrule(
    name = "simple",
    srcs = glob(["src/**/*.zig", "build.zig", "build.zig.zon"]),
    outs = ["zig-out/bin/simple"],
    cmd = "ZG_DATA_DIR=$$(dirname $$(echo $(locations :mnist_data) | cut -d' ' -f1)) ZIG_LOCAL_CACHE_DIR=$$(pwd)/zig-cache ZIG_GLOBAL_CACHE_DIR=$$(pwd)/zig-cache zig build simple -Doptimize=ReleaseFast && cp zig-out/bin/simple $@",
    tools = [":mnist_data"],
)


genrule(
    name = "timing",
    srcs = glob(["src/**/*.zig", "build.zig", "build.zig.zon"]),
    outs = ["zig-out/bin/timing"],
    cmd = "ZG_DATA_DIR=$$(dirname $$(echo $(locations :mnist_data) | cut -d' ' -f1)) ZIG_LOCAL_CACHE_DIR=$$(pwd)/zig-cache ZIG_GLOBAL_CACHE_DIR=$$(pwd)/zig-cache zig build timing -Doptimize=ReleaseFast && cp zig-out/bin/timing $@",
    tools = [":mnist_data"],
)

# Locations returns a list, need just the dir of the first item.
# Only way to do this seems to be with a wrapper script. Someone please fix.
sh_binary(
    name = "simple_bin",
    srcs = ["_bzl_run_helper.sh"],
    data = [":mnist_data", ":simple"],
    args = ["$(location :simple)", "$(locations :mnist_data)"],
)

sh_binary(
    name = "timing_bin",
    srcs = ["_bzl_run_helper.sh"],
    data = [":mnist_data", ":timing"],
    args = ["$(location :timing)", "$(locations :mnist_data)"],
)

load("@rules_python//python:defs.bzl", "py_runtime", "py_binary")
load("@mnist_deps//:requirements.bzl", "requirement")

py_binary(
    name = "generate_test_data",
    srcs = ["mnist_data.py"],
    main = "mnist_data.py",
    deps = [
        requirement("torch"),
        requirement("torchvision"),
    ],
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
