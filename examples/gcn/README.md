# gcn

## download dataset

```bash
uv run ref/dataset.py
```

## train

```bash
# pytorch
uv run ref/train.py

# zigrad
zig build run -Doptimize=ReleaseFast
# if you have MKL (even if its your system BLAS this will enable more integration):
zig build run -Doptimize=ReleaseFast -Denable_mkl=true
```
