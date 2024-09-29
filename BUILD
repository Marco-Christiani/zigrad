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
    srcs = glob(["src/**/*.zig", "build.zig"]),
    outs = ["zigrad_lib.a"],  # Output library from build.zig
    cmd = "zig build -Drelease-safe && cp zig-out/lib/libzigrad.a $@",
)

load("@rules_python//python:defs.bzl", "py_runtime", "py_binary")
load("@rules_python//python:pip.bzl", "pip_install")

# Define a hermetic Python runtime (Python 3.8)
py_runtime(
    name = "python3_8_runtime",
    interpreter_path = "/path/to/python3.8/bin/python3",  # Adjust this to your Python 3.8 path
    files = [],  # You can leave this empty
)

# Install PyTorch and dependencies without requirements.txt
pip_install(
    name = "pip_deps",
    packages = {
        "torch": "1.9.0",
        "numpy": "1.19.2",  # Include any other necessary packages here
    },
)

# Create a py_binary rule for generating test data
py_binary(
    name = "generate_test_data",
    srcs = ["tests/generate_test_data.py"],  # Path to your Python script
    main = "tests/generate_test_data.py",    # Entry point for the Python script
    runtime = ":python3_8_runtime",          # Use the defined Python 3.8 runtime
    deps = [":pip_deps"],                    # Include PyTorch and numpy as dependencies
)
