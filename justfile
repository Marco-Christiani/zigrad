default: (br "src/tensor/conv_test.zig")
run_utils: (run_pattern "src/tensor/utils.zig")

test_zarray: (btest "src/tensor/zarray.zig")
test_tensor: (btest "src/tensor/tensor.zig")
test_layer: (btest "src/tensor/layer.zig")

btest file:
  zig build test -Dfile={{file}}

testf filter:
  zig test --test-filter {{filter}} src/{{filter}}.zig -freference-trace

test file:
  zig test {{file}} -freference-trace

br file:
  zig build run -Dfile={{file}}

pattern := 'error\(gpa\)*'
run_pattern file:
   zig build run -Dfile={{file}} 2>&1 \
   | rg --passthru --color always {{pattern}} 2>&1 \
   | tee /dev/tty \
   | rg --passthru --color always --count-matches {{pattern}}
