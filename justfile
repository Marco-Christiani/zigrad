set unstable

default:
    @just gcn

[script("bash")]
gcn +opts="--release=fast -Denable_mkl=true":
    set -e
    export ZG_DATA_DIR=$(realpath data)
    cd examples/gcn
    uv run ref/dataset.py
    uv run ref/train.py > torch_timing.json
    zig build {{opts}}
    zig-out/bin/main > zg_timing.json

alias b := build
alias bf := build-fast
alias bt := test
alias r := run

export ZIGRAD_BACKEND := env("ZIGRAD_BACKEND", "HOST")

test +opts="":
  python src/nn/tests/test_loss.py
  zig build test {{opts}}

build +opts="":
  zig build {{opts}}

build-fast +opts="":
  zig build -Doptimize=ReleaseFast {{opts}}

run +opts="":
  zig build run {{opts}}

cuda_compile:
  pushd src/cuda/ && \
   mkdir build && cd build && \
   cmake .. && make -j$(nproc) && \
   popd

export ZG_DATA_DIR := env("ZG_DATA_DIR", "examples/mnist/data")
mnist args="":
  @python examples/mnist/mnist_data.py
  @echo "Compiling zigrad mnist example"
  @cd examples/mnist && zig build -Doptimize=ReleaseFast {{args}}
  @echo "Running zigrad mnist example"
  examples/mnist/zig-out/bin/main

benchmark +verbose="":
  @python examples/mnist/mnist_data.py
  @echo "Running pytorch mnist"
  python src/nn/tests/test_mnist.py -t --batch_size=64 --num_epochs=3 --model_variant=simple \
    {{ if verbose != "" { "| tee" } else { ">" } }} /tmp/zg_mnist_torch_log.txt
  @echo "Compiling zigrad mnist"
  @cd examples/mnist && zig build -Doptimize=ReleaseFast #-Dtracy_enable=false
  @echo "Running zigrad mnist"
  examples/mnist/zig-out/bin/main 2>\
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

docker-br:
    docker build -t zigrad-cuda-unstable -f Dockerfile .
    docker run --rm --gpus all -it -v .:/app zigrad-cuda-unstable

pattern := '\[gpa\] \(err\)*'
run_pattern:
   zig build test 2>&1 \
   | tee /dev/tty \
   | python tb.py
   # | rg --passthru --color always --count-matches "{{pattern}}" 2>&1 \
   # | rg --passthru --color always --count-matches {{pattern}}
