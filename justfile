set unstable

default:
  @just --choose

alias b := build
alias bt := test
alias r := run

test +opts="":
  zig build test {{opts}}

build +opts="":
  zig build {{opts}}

run +opts="":
  zig build run {{opts}}

export ZG_DATA_DIR := env("ZG_DATA_DIR", "data")
benchmark +verbose="":
  @python examples/mnist/mnist_data.py
  @echo "Running pytorch mnist"
  # python src/nn/tests/test_mnist.py -t --batch_size=64 --num_epochs=3 --model_variant=simple \
  #   {{ if verbose != "" { "| tee" } else { ">" } }} /tmp/zg_mnist_torch_log.txt
  python src/nn/tests/test_mnist_tf.py -t --batch_size=64 --num_epochs=3 --model_variant=simple --device=cpu \
    {{ if verbose != "" { "| tee" } else { ">" } }} /tmp/zg_mnist_torch_log.txt
  @echo "Compiling zigrad mnist"
  @just build -Doptimize=ReleaseFast -Dtracy_enable=true
  @echo "Running zigrad mnist"
  ./zig-out/bin/zigrad 2>\
    {{ if verbose != "" { "&1 | tee" } else { "" } }} /tmp/zg_mnist_log.txt
  @echo "Comparing results"
  python scripts/mnist_compare.py

[script("bash")]
doc:
  start_file_watcher() {
    if command -v fswatch &>/dev/null; then
      echo "Using fswatch to monitor changes..."
      fswatch -o *.zig src | while read -r event
      do
        echo "Change detected, rebuilding docs..."
        zig build docs
      done &
    elif command -v inotifywait &>/dev/null; then
      echo "Using inotifywait to monitor changes..."
      inotifywait -m -e modify,create,delete path | while read -r directory events filename
      do
        echo "Change detected in $directory$filename, rebuilding docs..."
        zig build docs
      done &
    else
      echo "Error: Neither fswatch nor inotifywait is available."
      exit 1
    fi
    watcher_pid=$!
  }
  cleanup() {
    echo "Stopping file watcher and cleaning up..."
    kill $watcher_pid
    echo "Cleanup complete."
  }
  trap cleanup EXIT INT
  start_file_watcher
  cd ./zig-out/docs/ && python -m http.server

[group("examples")]
example-mnist:
  docker run -it -v $(pwd):/workspace -e DEBIAN_FRONTEND="noninteractive" debian bash -c "\
  apt-get update -y && apt-get install -y make curl xz-utils libopenblas-dev python3-minimal && \
  cd /workspace/examples/mnist/ && bash"

pattern := 'error\(gpa\)*'
run_pattern file:
   zig build run -Dfile={{file}} 2>&1 \
   | rg --passthru --color always {{pattern}} 2>&1 \
   | tee /dev/tty \
   | rg --passthru --color always --count-matches {{pattern}}
   # | rg --passthru --color always --count-matches {{pattern}} 2&1 \
   # | .venv/bin/python tb.py
