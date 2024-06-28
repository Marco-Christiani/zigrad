default: test_layer

test_zarray:
  zig build test -Dfile=src/tensor/zarray.zig

test_tensor:
  zig build test -Dfile=src/tensor/tensor.zig

test_layer:
  zig build test -Dfile=src/tensor/layer.zig


test filter:
  zig test --test-filter {{filter}} src/{{filter}}.zig -freference-trace
