"""
Experimentation in python to speed up finding answers and establishing understanding

Dumping objects, IR, etc, is useful for telling us where to look and determining actual runtime
behavior when it is complex to reason about from source code alone.

This is adapted as necessary. Beware of version issues between the python venv and the cloned
source code repo we are grepping. One option is to actually import the repo's version or install
the repo in the venv. This is not always applicable if we arent grepping the codebase but only
trying to understand how X product does Y.
"""

import argparse
import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jaxlib
import jax.numpy as jnp


class Program:
    NAME: str
    func: ClassVar[Callable[..., Any]]
    inputs: ClassVar[tuple[Any, ...]]

    @classmethod
    def ir(cls):
        return jax.make_jaxpr(cls.func)(*cls.inputs)

    @classmethod
    def stablehlo(cls):
        lowered = jax.jit(cls.func).lower(*cls.inputs)
        return lowered.compiler_ir(dialect="stablehlo")

    @classmethod
    def hlo(cls) -> str:
        lowered = jax.jit(cls.func).lower(*cls.inputs)
        return lowered.compiler_ir(dialect="hlo").as_hlo_text()

    @classmethod
    def display(cls, *, show: set[str]):
        chunks: list[str] = [cls.NAME]
        if "source" in show:
            chunks += ["Source:", inspect.getsource(cls.func).rstrip()]
        if "jaxpr" in show:
            chunks += ["JAXPR:", str(cls.ir())]
        if "stablehlo" in show:
            chunks += ["StableHLO:", str(cls.stablehlo()).rstrip()]
        if "hlo" in show:
            chunks += ["HLO:", cls.hlo().rstrip()]
        return "\n".join(chunks) + "\n"


class Program1(Program):
    NAME = "Basic elemenwise: x*y + 1"
    x = jnp.ones((4,))
    y = jnp.ones((4,))
    inputs = (x, y)

    @staticmethod
    def func(x, y):
        a = x * y
        b = a + 1.0
        return b


class Program2(Program):
    NAME = "Matmul: x@y (dot_general)"
    x = jnp.ones((2, 3), dtype=jnp.float32)
    y = jnp.ones((3, 4), dtype=jnp.float32)
    inputs = (x, y)

    @staticmethod
    def func(x, y):
        return x @ y


class Program3(Program):
    NAME = "Transpose: transpose(x, (1,0))"
    x = jnp.ones((2, 3), dtype=jnp.float32)
    inputs = (x,)

    @staticmethod
    def func(x):
        return jnp.transpose(x, (1, 0))


PROGRAMS: dict[str, type[Program]] = {
    "Program1": Program1,
    "Program2": Program2,
    "Program3": Program3,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference IR experiments (JAXPR / StableHLO / HLO).",
    )
    parser.add_argument(
        "--program",
        default="all",
        choices=["all", *sorted(PROGRAMS.keys())],
        help="Which program to run.",
    )
    parser.add_argument(
        "--show",
        default="source,jaxpr,stablehlo",
        help="Comma-separated: source,jaxpr,stablehlo,hlo",
    )
    args = parser.parse_args()

    show = {s.strip() for s in args.show.split(",") if s.strip()}
    unknown = sorted(show - {"source", "jaxpr", "stablehlo", "hlo"})
    if unknown:
        raise SystemExit(f"Unknown --show entries: {', '.join(unknown)}")

    print(f"jax={jax.__version__} jaxlib={jaxlib.__version__}")
    print(f"backend={jax.default_backend()} devices={[d.platform for d in jax.devices()]}")
    print()

    if args.program == "all":
        for prog in sorted(PROGRAMS.values(), key=lambda p: p.NAME):
            print(prog.display(show=show))
    else:
        print(PROGRAMS[args.program].display(show=show))


if __name__ == "__main__":
    main()
