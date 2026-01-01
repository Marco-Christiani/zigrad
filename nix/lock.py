"""Lockgen

Goal:
  Produce a single, canonical lock file (nix/lock.json) that captures:
    1) the resolved wheel set (urls + hashes) for a given python tag + platform
    2) the derived provenance pins (JAX git hash -> XLA commit -> LLVM/StableHLO commits)

Rationale:
  - uv.lock already contains wheel URLs and sha256 (hex).
  - but "true lock" for rebuilding headers/SDK requires deriving the source commits too.
  - nix is best when given fixed inputs, uv is best for solving. This script bridges them.

Notes:
  - uv.lock hashes are "sha256:<hex>". Nix prefers SRI ("sha256-<base64>").
  - We do the conversion here once so Nix never needs ad-hoc mapping tables.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import json
import logging
import platform
import re
import sys
import sysconfig
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Wheel:
    pkg: str
    version: str
    url: str
    sha256_hex: str
    hash_sri: str


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
        s_l = s.lower()
        if "x86_64" in s_l or "amd64" in s_l:
            return cls.x86_64
        if "aarch64" in s_l or "arm64" in s_l:
            return cls.aarch64
        logger.warning(
            "Didnt recognize arch in string '%s', returning `Arch.none` good luck (could cause a failure downstream, prob fix me).",
            s,
        )
        return cls.none


def _hex_sha256_to_sri(hex_sha256: str) -> str:
    """Convert 64-hex sha256 into SRI string accepted by nix fetchers."""
    h = hex_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise ValueError(f"Expected sha256 hex (64 chars), got: {hex_sha256!r}")
    raw = bytes.fromhex(h)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"sha256-{b64}"


def _is_linux_wheel(base: str) -> bool:
    base_l = base.lower()
    if "win_" in base_l or "macosx" in base_l:
        return False
    return ("manylinux" in base_l) or ("musllinux" in base_l)


def _wheel_matches(url: str, *, py_tag: str, plat: str) -> bool:
    # Match wheel filename tags in the URL. We only need a robust-enough heuristic.
    # Example:
    #   .../jax_cuda13_plugin-0.8.3.dev...-cp314-cp314-manylinux_2_27_x86_64.whl
    #   .../jax_cuda13_pjrt-0.8.3.dev...-py3-none-manylinux_2_27_x86_64.whl
    base = url.rsplit("/", 1)[-1]
    if not base.endswith(".whl"):
        return False

    base_l = base.lower()
    plat_l = plat.lower()

    # get requested platform family
    if "manylinux" in plat_l:
        target_family = "manylinux"
    elif "musllinux" in plat_l:
        target_family = "musllinux"
    else:
        # strict. if caller gives something else, require it appear in filename.
        target_family = None

    if not _is_linux_wheel(base):
        return False

    # enforce platform family match (prevents manylinux vs musllinux mixing)
    if target_family is not None:
        if target_family not in base_l:
            return False
        if target_family == "manylinux" and "musllinux" in base_l:
            return False
        if target_family == "musllinux" and "manylinux" in base_l:
            return False

    target_arch = Arch.parse(plat)
    wheel_arch = Arch.parse(base)
    if wheel_arch != target_arch:
        return False

    # pure wheels: accept any linux wheel for correct arch
    if "-py3-none-" in base:
        return True

    # abi-bound extension wheels (cp*), require exact python tag
    return f"-{py_tag}-" in base


def _default_py_tag() -> str:
    """Best-effort default CPython tag for wheels.

    Examples:
      cp314-cp314
      cp314-cp314t   (free-threaded)
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    impl = platform.python_implementation().lower()
    if impl != "cpython":
        # keep it explicit. user can override.
        return f"py{major}-none"

    base = f"cp{major}{minor}"
    # detect free-threaded builds (3.13+ typically) when available
    # this is intentionally heuristic to avoid external deps
    abi_flags = str(sysconfig.get_config_var("ABIFLAGS") or "")
    gil_disabled = str(sysconfig.get_config_var("Py_GIL_DISABLED") or "")
    is_ft = ("t" in abi_flags) or (gil_disabled == "1")
    if is_ft:
        return f"{base}-{base}t"
    return f"{base}-{base}"


def _default_plat() -> str:
    """Best-effort default linux platform tag for the current arch.

    We default to manylinux_2_27_* because that's what the JAX CUDA wheels commonly target.
    Override with --plat if needed.
    """
    m = platform.machine().lower()
    arch = Arch.parse(m)
    if arch == Arch.x86_64:
        return "manylinux_2_27_x86_64"
    if arch == Arch.aarch64:
        return "manylinux_2_27_aarch64"
    # fallback
    return "manylinux_2_27_x86_64"


def _http_get_text(url: str, *, timeout_s: float = 30.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "zigrad-lockgen/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
    return data.decode("utf-8", errors="replace")


def _download(url: str, out: Path, *, timeout_s: float = 60.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "zigrad-lockgen/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        out.write_bytes(r.read())


def _find_or_fetch_wheel(wheel: Wheel, *, cache_dir: Path) -> Path:
    """Locate the wheel in a local cache if possible, otherwise download it.

    We do not assume uv's internal cache layout (it can change).
    We simply search for the wheel filename under cache_dir, and download if not found.
    """
    filename = wheel.url.rsplit("/", 1)[-1]
    # fast path: search for filename
    for p in cache_dir.rglob(filename):
        if p.is_file():
            return p

    # download to a stable location
    dst = cache_dir / "downloads" / filename
    logger.warning("Wheel %s not found in cache; downloading to %s", filename, dst)
    _download(wheel.url, dst)
    return dst


def _extract_jax_git_hash_from_plugin_wheel(wheel_path: Path) -> str:
    """Extract JAX git hash embedded in jax-cuda*-plugin wheel's version.py.

    This is how we turn 'wheels' into a source-level pin.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        # find version.py in the plugin package
        candidates = [n for n in zf.namelist() if n.endswith("/version.py")]
        if not candidates:
            raise RuntimeError(f"No version.py found in wheel: {wheel_path}")
        # prefer jax_cuda*_plugin/version.py if present
        preferred = [n for n in candidates if "jax_cuda" in n and "_plugin/" in n]
        name = preferred[0] if preferred else candidates[0]
        txt = zf.read(name).decode("utf-8", errors="replace")

    m = re.search(r"_git_hash:\s*str\s*=\s*'([0-9a-f]{40})'", txt)
    if not m:
        raise RuntimeError(
            f"Could not find _git_hash in {wheel_path} ({name}). "
            "Wheel may omit git hash; pass --jax-git-hash to override."
        )
    return m.group(1)


@dataclass(frozen=True)
class DerivedPins:
    jax_git_hash: str
    xla_commit: str
    xla_sha256_hex: str
    xla_sha256_sri: str
    xla_tarball_url: str

    llvm_commit: str
    llvm_sha256_hex: str
    llvm_sha256_sri: str
    llvm_urls: list[str]

    stablehlo_commit: str
    stablehlo_sha256_hex: str
    stablehlo_sha256_sri: str
    stablehlo_urls: list[str]


def _parse_bzl_assign(text: str, var: str) -> str:
    # simple and intentionally strict: VAR = "..."
    pat = rf'^{re.escape(var)}\s*=\s*"([^"]+)"\s*$'
    for line in text.splitlines():
        m = re.match(pat, line.strip())
        if m:
            return m.group(1)
    raise RuntimeError(f"Failed to parse {var} from bzl text")


def derive_pins_from_wheels(
    *,
    wheels: list[Wheel],
    cache_dir: Path,
    jax_git_hash_override: str | None,
) -> DerivedPins:
    # find the plugin wheel we can inspect for the embedded git hash
    plugin_candidates = [w for w in wheels if re.search(r"jax-cuda\d+-plugin", w.pkg)]
    if not plugin_candidates:
        raise RuntimeError(
            "No jax-cuda*-plugin wheel selected; cannot derive JAX git hash"
        )
    plugin_wheel = plugin_candidates[0]

    if jax_git_hash_override is not None:
        jax_git_hash = jax_git_hash_override
    else:
        wheel_path = _find_or_fetch_wheel(plugin_wheel, cache_dir=cache_dir)
        jax_git_hash = _extract_jax_git_hash_from_plugin_wheel(wheel_path)

    # derive XLA commit from JAX third_party pin.
    # dont need to clone we can fetch revision.bzl by commit hash.
    revision_bzl_url = f"https://raw.githubusercontent.com/jax-ml/jax/{jax_git_hash}/third_party/xla/revision.bzl"
    revision_bzl = _http_get_text(revision_bzl_url)
    xla_commit = _parse_bzl_assign(revision_bzl, "XLA_COMMIT")
    xla_sha256_hex = _parse_bzl_assign(revision_bzl, "XLA_SHA256").lower()
    xla_sha256_sri = _hex_sha256_to_sri(xla_sha256_hex)

    # JAX documents the hash as:
    #   curl -L https://api.github.com/repos/openxla/xla/tarball/{git_hash} | sha256sum
    xla_tarball_url = f"https://api.github.com/repos/openxla/xla/tarball/{xla_commit}"

    # derive LLVM/StableHLO pins from pinned XLA commit
    llvm_ws_url = f"https://raw.githubusercontent.com/openxla/xla/{xla_commit}/third_party/llvm/workspace.bzl"
    stablehlo_ws_url = f"https://raw.githubusercontent.com/openxla/xla/{xla_commit}/third_party/stablehlo/workspace.bzl"
    llvm_ws = _http_get_text(llvm_ws_url)
    stablehlo_ws = _http_get_text(stablehlo_ws_url)

    llvm_commit = _parse_bzl_assign(llvm_ws, "LLVM_COMMIT")
    llvm_sha256_hex = _parse_bzl_assign(llvm_ws, "LLVM_SHA256").lower()
    llvm_sha256_sri = _hex_sha256_to_sri(llvm_sha256_hex)

    # match XLA's actual fetch URLs
    llvm_urls = [
        f"https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{llvm_commit}.tar.gz",
        f"https://github.com/llvm/llvm-project/archive/{llvm_commit}.tar.gz",
    ]

    stablehlo_commit = _parse_bzl_assign(stablehlo_ws, "STABLEHLO_COMMIT")
    stablehlo_sha256_hex = _parse_bzl_assign(stablehlo_ws, "STABLEHLO_SHA256").lower()
    stablehlo_sha256_sri = _hex_sha256_to_sri(stablehlo_sha256_hex)
    stablehlo_urls = [
        f"https://github.com/openxla/stablehlo/archive/{stablehlo_commit}.zip",
    ]

    return DerivedPins(
        jax_git_hash=jax_git_hash,
        xla_commit=xla_commit,
        xla_sha256_hex=xla_sha256_hex,
        xla_sha256_sri=xla_sha256_sri,
        xla_tarball_url=xla_tarball_url,
        llvm_commit=llvm_commit,
        llvm_sha256_hex=llvm_sha256_hex,
        llvm_sha256_sri=llvm_sha256_sri,
        llvm_urls=llvm_urls,
        stablehlo_commit=stablehlo_commit,
        stablehlo_sha256_hex=stablehlo_sha256_hex,
        stablehlo_sha256_sri=stablehlo_sha256_sri,
        stablehlo_urls=stablehlo_urls,
    )


def collect_wheels(
    uv_lock: Path,
    *,
    roots: set[str],
    py_tag: str,
    plat: str,
) -> list[Wheel]:
    data = tomllib.loads(uv_lock.read_text(encoding="utf-8"))
    pkgs = {p["name"]: p for p in data["package"]}

    seen: set[str] = set()
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
            logger.debug("%s->%s", pkg["name"], dep["name"])
            queue.append(dep["name"])

        for deps in pkg.get("optional-dependencies", {}).values():
            for dep in deps:
                if dep["name"] in pkgs:
                    queue.append(dep["name"])

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
            if _wheel_matches(url, py_tag=py_tag, plat=plat):
                wheels.append(
                    Wheel(
                        pkg=name,
                        version=version,
                        url=url,
                        sha256_hex=sha256_hex,
                        hash_sri=_hex_sha256_to_sri(sha256_hex),
                    )
                )

    wheels.sort(key=lambda x: (x.pkg, x.version, x.url))
    return wheels


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate nix/lock.json from uv.lock (wheels + derived JAX->XLA->LLVM/StableHLO pins)."
    )
    ap.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    ap.add_argument("--out", type=Path, default=Path("nix/lock.json"))
    ap.add_argument(
        "--py-tag",
        type=str,
        default=_default_py_tag(),
        help="Wheel python tag (default: inferred from current interpreter). Example: cp314-cp314",
    )
    ap.add_argument(
        "--plat",
        type=str,
        default=_default_plat(),
        help="Wheel platform tag (default: inferred from machine arch). Example: manylinux_2_27_x86_64",
    )

    ap.add_argument(
        "--roots",
        nargs="+",
        default=["jax"],
        help=(
            "Root packages to traverse in uv.lock. Default: jax. "
            "If your uv.lock doesn't include 'jax' as a package name, pass explicit roots."
        ),
    )

    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("~/.cache/zigrad-lockgen").expanduser(),
        help="Local cache for wheel inspection downloads (used to extract JAX git hash).",
    )
    ap.add_argument(
        "--jax-git-hash",
        type=str,
        default=None,
        help="Override JAX git hash instead of extracting from the jax-cuda*-plugin wheel.",
    )

    ap.add_argument("--verbose", "-v", action="store_true", help="verbose")
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    wheels = collect_wheels(
        args.uv_lock,
        roots=set(args.roots),
        py_tag=str(args.py_tag),
        plat=str(args.plat),
    )
    if not wheels:
        raise SystemExit(
            f"No wheels selected (py_tag={args.py_tag!r}, plat={args.plat!r}, roots={args.roots!r})."
        )

    pins = derive_pins_from_wheels(
        wheels=wheels,
        cache_dir=args.cache_dir,
        jax_git_hash_override=args.jax_git_hash,
    )

    lock = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "selectors": {
            "py_tag": args.py_tag,
            "plat": args.plat,
            "roots": list(args.roots),
        },
        "wheels": [dataclasses.asdict(w) for w in wheels],
        "pins": {
            "jax": {
                "git_hash": pins.jax_git_hash,
                # TODO: add tag
                # not guaranteed to exist for dev builds, but helpful
                "tag_hint": None,
            },
            "xla": {
                "commit": pins.xla_commit,
                "tarball_url": pins.xla_tarball_url,
                "sha256_hex": pins.xla_sha256_hex,
                "hash_sri": pins.xla_sha256_sri,
            },
            "llvm": {
                "commit": pins.llvm_commit,
                "urls": pins.llvm_urls,
                "sha256_hex": pins.llvm_sha256_hex,
                "hash_sri": pins.llvm_sha256_sri,
            },
            "stablehlo": {
                "commit": pins.stablehlo_commit,
                "urls": pins.stablehlo_urls,
                "sha256_hex": pins.stablehlo_sha256_hex,
                "hash_sri": pins.stablehlo_sha256_sri,
            },
        },
    }

    _write_json(args.out, lock)
    logger.warning("Wrote %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    main()
