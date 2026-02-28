import argparse
from pathlib import Path
import sqlite3

import polars as pl


def load_kernels(sqlite_path: Path) -> pl.DataFrame:
    query = """
    select s.value as name, sum(k.end - k.start) as total_ns, count(*) as instances
    from CUPTI_ACTIVITY_KIND_KERNEL k
    join StringIds s on s.id = k.demangledName
    group by s.value
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        cur = con.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    finally:
        con.close()

    df = pl.DataFrame(rows, schema=cols, orient="row")
    if df.is_empty():
        return df.with_columns(
            pl.lit(0.0).alias("total_ms"),
            pl.lit(0.0).alias("pct"),
        )

    total = int(df["total_ns"].sum())
    return df.with_columns(
        (pl.col("total_ns") / 1_000_000).alias("total_ms"),
        (
            pl.when(pl.lit(total) == 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("total_ns") / pl.lit(total) * 100.0)
        ).alias("pct"),
    )


def build_table(zig_path: Path, jax_path: Path, top_n: int) -> pl.DataFrame:
    zig = load_kernels(zig_path).rename(
        {
            "total_ns": "zig_ns",
            "total_ms": "zig_ms",
            "pct": "zig_pct",
            "instances": "zig_instances",
        }
    )
    jax = load_kernels(jax_path).rename(
        {
            "total_ns": "jax_ns",
            "total_ms": "jax_ms",
            "pct": "jax_pct",
            "instances": "jax_instances",
        }
    )
    merged = zig.join(jax, on="name", how="full", suffix="_jax")
    merged = merged.with_columns(
        pl.coalesce([pl.col("name"), pl.col("name_jax")]).alias("kernel"),
        pl.coalesce([pl.col("zig_ns"), pl.lit(0)]).alias("zig_ns"),
        pl.coalesce([pl.col("jax_ns"), pl.lit(0)]).alias("jax_ns"),
        pl.coalesce([pl.col("zig_ms"), pl.lit(0.0)]).alias("zig_ms"),
        pl.coalesce([pl.col("jax_ms"), pl.lit(0.0)]).alias("jax_ms"),
        pl.coalesce([pl.col("zig_pct"), pl.lit(0.0)]).alias("zig_pct"),
        pl.coalesce([pl.col("jax_pct"), pl.lit(0.0)]).alias("jax_pct"),
    )
    merged = merged.with_columns(
        (pl.col("zig_ns") + pl.col("jax_ns")).alias("combined_ns")
    )
    return merged.sort("combined_ns", descending=True).head(top_n)


def to_markdown(df: pl.DataFrame) -> str:
    cols = ["kernel", "zig_ms", "zig_pct", "jax_ms", "jax_pct"]
    df = df.select(cols).with_columns(
        pl.col("zig_ms").round(3),
        pl.col("jax_ms").round(3),
        pl.col("zig_pct").round(1),
        pl.col("jax_pct").round(1),
    )
    lines = [
        "| Kernel | Zig ms | Zig % | JAX ms | JAX % |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in df.iter_rows():
        name, zig_ms, zig_pct, jax_ms, jax_pct = row
        lines.append(
            f"| {name} | {zig_ms:.3f} | {zig_pct:.1f} | {jax_ms:.3f} | {jax_pct:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned kernel tables from nsys sqlite exports."
    )
    parser.add_argument(
        "--zig", required=True, type=Path, help="Path to Zig .sqlite export"
    )
    parser.add_argument(
        "--jax", required=True, type=Path, help="Path to JAX .sqlite export"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Top N kernels by combined time"
    )
    parser.add_argument("--out", required=True, type=Path, help="Output markdown file")
    args = parser.parse_args()

    table = build_table(args.zig, args.jax, args.top)
    args.out.write_text(to_markdown(table))


if __name__ == "__main__":
    main()
