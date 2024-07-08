default: bt

bt:
  zig build test -Dlog_level=debug

b:
  zig build

btest file:
  zig build test -Dfile={{file}}

testf filter:
  zig test --test-filter {{filter}} src/{{filter}}.zig -freference-trace

test file:
  zig test {{file}} -freference-trace

btz:
  @just (btest "src/tensor/zarray.zig")

btt:
  @just (btest "src/tensor/tensor.zig")

btl:
  @just (btest "src/tensor/layer.zig")

br opts="":
  zig build run {{opts}}

brm:
  @just br src/tensor/mnist.zig

brc:
  @just (br "src/tensor/conv_test.zig")

pattern := 'error\(gpa\)*'
run_pattern file:
   zig build run -Dfile={{file}} 2>&1 \
   | rg --passthru --color always {{pattern}} 2>&1 \
   | tee /dev/tty \
   | rg --passthru --color always --count-matches {{pattern}}
   # | rg --passthru --color always --count-matches {{pattern}} 2&1 \
   # | .venv/bin/python tb.py


brpu:
  @just (run_pattern "src/tensor/utils.zig")
