from __future__ import annotations

import argparse
import logging
import re
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Wheel:
    pkg: str
    version: str
    url: str
    sha256: str


class Arch(StrEnum):
    x86_64 = auto()
    aarch64 = auto()
    none = auto()

    @classmethod
    def parse(cls, s: str) -> Arch:
        """Extract cpu arch from platform specifier (or a wheel name or similar).

        plat: proper platform specifier or something close,
            e.g. manylinux_2_27_x86_64 or manylinux_2_27_aarch64 or foo_aarch64.whl

        Some other things that exist that I havnt thought about (ofc the version numbers can vary):
          macosx_11_0_arm64
          win_amd64
          win_arm64
          macosx_10_13_universal2
          musllinux_1_2_aarch64
          musllinux_1_2_x86_64
        """
        s = s.lower()
        if "x86_64" in s or "amd64" in s:
            return cls.x86_64
        if "aarch64" in s or "arm64" in s:
            return cls.aarch64
        logger.warning(
            "Didnt recognize arch in string '%s', returning `Arch.none` good luck (could cause a failure downstream, prob fix me).",
            s,
        )
        return cls.none


def _is_linux_wheel(base: str) -> bool:
    base_l = base.lower()
    if "win_" in base_l or "macosx" in base_l:
        return False
    return ("manylinux" in base_l) or ("musllinux" in base_l)


def _wheel_matches(url: str, *, py_tag: str, plat: str) -> bool:
    # Match wheel filename tags in the URL. We only need a robust-enough heuristic.
    # Example:
    #   .../jax_cuda12_plugin-0.8.3.dev...-cp314-cp314-manylinux_2_27_x86_64.whl
    #   .../jax_cuda12_pjrt-0.8.3.dev...-py3-none-manylinux_2_27_x86_64.whl
    base = url.rsplit("/", 1)[-1]
    if not base.endswith(".whl"):
        return False

    # Determine target arch from the requested platform specifier
    try:
        target_arch = Arch.parse(plat)
    except ValueError:
        # plat should always encode arch for our use
        raise

    if not _is_linux_wheel(base):
        return False

    # Wheel arch must match target arch
    try:
        wheel_arch = Arch.parse(base)
    except ValueError:
        return False
    if wheel_arch != target_arch:
        return False

    # Pure wheels: accept any linux wheel for correct arch
    if "-py3-none-" in base:
        return True

    # abi-bound extension wheels (cp*), require exact python tag
    return f"-{py_tag}-" in base


def collect_wheels(
    uv_lock: Path,
    *,
    roots: set[str],
    py_tag: str,
    plat: str,
) -> list[Wheel]:
    data = tomllib.loads(uv_lock.read_text(encoding="utf-8"))
    pkgs = {p["name"]: p for p in data["package"]}

    seen = set()
    queue = list(roots)

    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)

        pkg = pkgs.get(name)
        if not pkg:
            continue

        for dep in pkg.get("dependencies", []):
            logger.debug(f"{pkg['name']}->{dep['name']}")
            queue.append(dep["name"])

        for deps in pkg.get("optional-dependencies", {}).values():
            for dep in deps:
                if dep["name"] in pkgs:
                    queue.append(dep["name"])
    for i, e in enumerate(seen):
        logger.debug(f"seen {i}: {e}")
    wheels: list[Wheel] = []
    for name in seen:
        pkg = pkgs[name]
        version = str(pkg.get("version", ""))
        for w in pkg.get("wheels", []):
            url = str(w["url"])
            h = str(w["hash"])
            # uv.lock uses "sha256:<hex>"
            m = re.fullmatch(r"sha256:([0-9a-fA-F]{64})", h)
            if m is None:
                raise ValueError(f"Unexpected hash format: {h} for {name}")
            sha256_hex = m.group(1).lower()
            if _wheel_matches(w["url"], py_tag=py_tag, plat=plat):
                wheels.append(
                    Wheel(pkg=name, version=version, url=url, sha256=sha256_hex)
                )

    # stable ordering
    wheels.sort(key=lambda x: (x.pkg, x.version, x.url))
    return wheels


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
            "jax-cuda13-pjrt",
            "jax-cuda13-plugin",
        ],
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="verbose")
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    wheels = collect_wheels(
        args.uv_lock,
        roots=set(args.pkgs),
        py_tag=str(args.py_tag),
        plat=str(args.plat),
    )
    for w in wheels:
        logger.debug(w.pkg)
    print(to_nix_fetchurls(wheels), end="")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    main()
