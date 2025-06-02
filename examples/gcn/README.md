# gcn

## download dataset

```bash
uv venv  # or other venv tool
uv pip install -r ref/requirements.txt
uv run python -- ref/dataset.py
```

## train

```bash
# pytorch
uv run python -- ref/train.py

# zigrad
export ZIGRAD_BACKEND=HOST
zig build run
```
