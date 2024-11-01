# vim:commentstring=#\ %s
# Setting up the venv is slow so just do it before running the tape
# Thats also why we have the tmp dir thing going on

Output zigrad-demo.mp4
Set FontSize 12

Hide
# Real dir where we run the demo (venv already set up)
# Use tmp dir to show cloning
Type "cd .. && pushd $(mktemp -d) && clear"
Enter
Show

Type "git clone https://github.com/Marco-Christiani/zigrad/"
Enter
Sleep 1.5
Type "cd zigrad"
Sleep 0.8
Enter
Type "clear"
Sleep 0.5
Enter

# Back to real dir
Hide
Type "popd && pwd && clear"
Enter
Show

Type "cd examples/mnist"
Sleep 0.8
Enter

Type "make help"
Sleep 0.8
Enter
Sleep 3

Type "clear"
Sleep 0.5
Enter

Type "make"
Sleep 0.8
Enter
Sleep 7

Type "clear"
Sleep 0.5
Enter
Type "# Run benchmarks"
Enter
Type "just benchmark -v"
Enter
Sleep 15

## This is just too slow tbh maybe save in a different tape
# Type "# Or, if you prefer bazel for better isolation..."
# Sleep 1
# Enter
# Type "bazel query //..."
# Enter
# Sleep 3
# Type "bazel build //:build"
# Sleep 10
# Enter
# Type "bazel-out/bin/timing"
# Enter
# Sleep 3
