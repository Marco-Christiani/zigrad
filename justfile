default: bt

bt +opts="":
  zig build test

b:
  zig build

btest file:
  zig build test -Dfile={{file}}

testf filter:
  zig test --test-filter {{filter}} src/{{filter}}.zig -freference-trace

test file:
  zig test {{file}} -freference-trace

btz:
  @just (btest "src/ndarray.zig")

btt:
  @just (btest "src/ndtensor.zig")

btl:
  @just (btest "src/nn/layer.zig")

br +opts="":
  zig build run {{opts}}

brm:
  @just br src/nn/mnist.zig

brc:
  @just (br "src/nn/conv_test.zig")

pattern := 'error\(gpa\)*'
run_pattern file:
   zig build run -Dfile={{file}} 2>&1 \
   | rg --passthru --color always {{pattern}} 2>&1 \
   | tee /dev/tty \
   | rg --passthru --color always --count-matches {{pattern}}
   # | rg --passthru --color always --count-matches {{pattern}} 2&1 \
   # | .venv/bin/python tb.py

brpu:
  @just (run_pattern "src/nn/utils.zig")

benchmark:
  ZG_DATA_DIR=/tmp/zgmnist python examples/mnist/mnist_data.py
  ZG_DATA_DIR=/tmp/zgmnist python src/nn/tests/test_mnist.py -t --batch_size=64 --num_epochs=3 --model_variant=simple > /tmp/zg_mnist_torch_log.txt
  ZG_DATA_DIR=/tmp/zgmnist just br -Doptimize=ReleaseFast -Dtracy_enable=false 2> /tmp/zg_mnist_log.txt
  python scripts/mnist_compare.py

doc:
  zig build docs
  cd ./zig-out/docs/ && python -m http.server
