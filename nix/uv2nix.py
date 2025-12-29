from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import logging
import tomllib
import sys

VERBOSE = sys.stdin.isatty()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Wheel:
    pkg: str
    version: str
    url: str
    sha256: str


def _wheel_matches(url: str, *, py_tag: str, plat: str) -> bool:
    # Match wheel filename tags in the URL. We only need a robust-enough heuristic.
    # Example:
    #   .../jax_cuda12_plugin-0.8.3.dev...-cp314-cp314-manylinux_2_27_x86_64.whl
    #   .../jax_cuda12_pjrt-0.8.3.dev...-py3-none-manylinux_2_27_x86_64.whl
    base = url.rsplit("/", 1)[-1]
    if not base.endswith(".whl"):
        return False

    # Require platform tag
    if plat not in base:
        return False

    # Require python tag for "cp..." wheels, allow py3-none wheels regardless of py_tag.
    if "-py3-none-" in base:
        return True

    return f"-{py_tag}-" in base


def collect_wheels(
    uv_lock: Path,
    *,
    wanted_pkgs: set[str],
    py_tag: str,
    plat: str,
) -> list[Wheel]:
    data = tomllib.loads(uv_lock.read_text(encoding="utf-8"))
    out: list[Wheel] = []

    for pkg in data.get("package", []):
        name = str(pkg.get("name", ""))
        if name == "jax-cuda13-plugin":
            od = pkg["optional-dependencies"]
            odn = {e["name"] for e in od["with-cuda"]}
            logger.debug(f"{set(wanted_pkgs) - odn} {odn - set(wanted_pkgs)}")
        if name not in wanted_pkgs and not name.startswith("nvidia-"):
            logger.debug("skipping pkg.name: %s", name)
            continue

        version = str(pkg.get("version", ""))

        for w in pkg.get("wheels", []):
            url = str(w["url"])
            h = str(w["hash"])
            # uv.lock uses "sha256:<hex>"
            m = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", h)
            if m is None:
                raise ValueError(f"Unexpected hash format: {h} for {name}")
            sha256_hex = m.group(1).lower()

            if _wheel_matches(url, py_tag=py_tag, plat=plat):
                out.append(Wheel(pkg=name, version=version, url=url, sha256=sha256_hex))

    # Stable ordering
    out.sort(key=lambda x: (x.pkg, x.version, x.url))
    return out


def to_nix_fetchurls(wheels: list[Wheel]) -> str:
    # Emit as:
    # {
    #   "jax-cuda12-plugin" = [
    #     { url = "..."; sha256 = "..."; }
    #   ];
    # }
    lines: list[str] = []
    lines.append("{")
    by_pkg: dict[str, list[Wheel]] = {}
    for w in wheels:
        by_pkg.setdefault(w.pkg, []).append(w)

    for pkg in sorted(by_pkg):
        lines.append(f'  "{pkg}" = [')
        for w in by_pkg[pkg]:
            lines.append("    {")
            lines.append(f'      url = "{w.url}";')
            lines.append(f'      sha256 = "{w.sha256}";')
            lines.append("    }")
        lines.append("  ];")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    ap.add_argument(
        "--py-tag", required=True, help="e.g. cp314-cp314 or cp314-cp314t"
    )  # can we avoid this?
    ap.add_argument(
        "--plat",
        required=True,
        help="e.g. manylinux_2_27_x86_64 or manylinux_2_27_aarch64",
    )  # can we avoid this?
    ap.add_argument(  # is harcoding the right idea? should at minimum make cuda version configurable
        "--pkgs",
        nargs="+",
        default=[
            # dont think we need these
            # "jax",
            # "jaxlib",
            "jax-cuda13-pjrt",
            "jax-cuda13-plugin",
            # TODO: The proper approach is to follow the delcared dependencies...
            #   Anyways, you can find this list under jax-cuda13-plugin package.optional-dependencies.with-cuda
            "nvidia-cublas",
            "nvidia-cuda-cupti",
            "nvidia-cuda-nvrtc",
            "nvidia-cuda-nvcc",
            "nvidia-cuda-runtime",
            "nvidia-cudnn-cu13",
            "nvidia-cufft",
            "nvidia-cusolver",
            "nvidia-cusparse",
            "nvidia-nccl-cu13",
            "nvidia-nvjitlink",
            "nvidia-nvshmem-cu13",
        ],
    )
    args = ap.parse_args()

    wheels = collect_wheels(
        args.uv_lock,
        wanted_pkgs=set(args.pkgs),
        py_tag=str(args.py_tag),
        plat=str(args.plat),
    )
    print(to_nix_fetchurls(wheels), end="")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    main()
