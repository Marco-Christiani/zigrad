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
  python src/nn/tests/test_mnist.py -t --batch_size=64 --num_epochs=3 --model_variant=simple \
    {{ if verbose != "" { "| tee" } else { ">" } }} /tmp/zg_mnist_torch_log.txt
  @echo "Compiling zigrad mnist"
  @just build -Doptimize=ReleaseFast -Dtracy_enable=false
  @echo "Running zigrad mnist"
  ./zig-out/bin/zigrad 2>\
    {{ if verbose != "" { "&1 | tee" } else { "" } }} /tmp/zg_mnist_log.txt
  @echo "Comparing results"
  python scripts/mnist_compare.py

doc:
  zig build docs
  cd ./zig-out/docs/ && python -m http.server

pattern := 'error\(gpa\)*'
run_pattern file:
   zig build run -Dfile={{file}} 2>&1 \
   | rg --passthru --color always {{pattern}} 2>&1 \
   | tee /dev/tty \
   | rg --passthru --color always --count-matches {{pattern}}
   # | rg --passthru --color always --count-matches {{pattern}} 2&1 \
   # | .venv/bin/python tb.py
