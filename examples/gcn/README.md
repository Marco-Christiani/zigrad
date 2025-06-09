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
export ZIGRAD_BACKEND=HOST
zig build run
```
