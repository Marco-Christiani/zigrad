#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyelftools>=0.32",
# ]
# ///

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn

from elftools.elf.elffile import ELFFile

LOG = logging.getLogger("elf-deps")


SYSTEM_ALLOWLIST = {
    "linux-vdso.so",
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "librt.so.1",
    "libpthread.so.0",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "libcuda.so.1",
}


@dataclass(frozen=True)
class FoundDep:
    soname: str
    search_dir: Path
    full_path: Path


def die(msg: str) -> NoReturn:
    LOG.error(msg)
    raise SystemExit(1)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def resolve_sdk_root(cli_value: Path | None) -> tuple[Path, str]:
    if cli_value is not None:
        if cli_value.exists():
            return cli_value, "cli"
        die(f"--sdk-root does not exist: {cli_value}")

    env = os.environ.get("ZG_EXTERNAL_SDK_ROOT")
    if env:
        root = Path(env)
        if root.exists():
            return root, "env:ZG_EXTERNAL_SDK_ROOT"
        die(f"ZG_EXTERNAL_SDK_ROOT is set but does not exist: {root}")

    die("No SDK root found (use --sdk-root or set ZG_EXTERNAL_SDK_ROOT)")


def read_elf_deps(path: Path) -> tuple[list[str], list[str]]:
    needed: list[str] = []
    rpaths: list[str] = []

    with path.open("rb") as f:
        elf = ELFFile(f)
        dyn = elf.get_section_by_name(".dynamic")
        if not dyn:
            return needed, rpaths

        for tag in dyn.iter_tags():
            if tag.entry.d_tag == "DT_NEEDED":
                needed.append(tag.needed)
            elif tag.entry.d_tag in ("DT_RPATH", "DT_RUNPATH"):
                value = getattr(tag, "runpath", None)
                if value:
                    rpaths.extend(value.split(":"))

    return needed, rpaths


def expand_origin(p: str, origin: Path) -> Path:
    return Path(p.replace("$ORIGIN", str(origin)).replace("${ORIGIN}", str(origin)))


def unique_dirs(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p.is_dir() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def classify(resolved: Path, sdk_root: Path) -> str:
    try:
        r = resolved.resolve()
        root = sdk_root.resolve()
    except Exception:
        return "OK"

    if r.is_relative_to(root / "lib"):
        return "OK lib"
    if r.is_relative_to(root / "runtime" / "sys" / "lib"):
        return "OK runtime"
    if str(r).startswith("/nix/store/"):
        return "OK nix"
    if r.is_absolute():
        return "OK sys"
    return "OK"


def display_path(
    dep: FoundDep,
    *,
    sdk_root: Path,
    origin: Path,
    expand_origin: bool,
    resolve_symlinks: bool,
) -> str:
    p = dep.full_path

    if resolve_symlinks:
        try:
            p = p.resolve()
        except Exception:
            pass

    if not expand_origin:
        try:
            rel = p.relative_to(origin)
            return f"$ORIGIN/{rel}"
        except Exception:
            pass

        try:
            rel = p.relative_to(sdk_root)
            return f"<sdk-root>/{rel}"
        except Exception:
            pass

    return str(p)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Inspect ELF DT_NEEDED dependencies and resolve them against an SDK layout.",
    )

    ap.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="ELF file to inspect (default: <sdk-root>/lib/libStablehloCAPI.so)",
    )

    ap.add_argument(
        "--sdk-root",
        type=Path,
        help="Path to SDK root dir (default: $ZG_EXTERNAL_SDK_ROOT)",
    )

    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--expand-origin", action="store_true")
    ap.add_argument("--resolve-symlinks", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument(
        "--allow-host",
        action="store_true",
        help="Allow host-provided system libs (glibc/driver) to satisfy deps.",
    )
    ap.add_argument(
        "--check-host",
        action="store_true",
        help="Search common host lib paths when resolving deps.",
    )
    ap.add_argument("--json", action="store_true")

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    configure_logging(args.verbose)

    sdk_root, sdk_root_source = resolve_sdk_root(args.sdk_root)
    lib_dir = sdk_root / "lib"
    runtime_sys = sdk_root / "runtime" / "sys" / "lib"

    target = args.target or (lib_dir / "libStablehloCAPI.so")
    target_source = "cli" if args.target else "<sdk-root>/lib default"
    target_abs = target.resolve()

    if not target.exists():
        if target.is_absolute() and target_abs == target:
            die(f"Target not found: {target}")
        else:
            die(f"Target not found: {target} (absolute path: {target_abs})")

    origin = target_abs.parent

    # Context
    LOG.info("SDK root:        %s (source: %s)", sdk_root, sdk_root_source)
    LOG.info("Target ELF:      %s (source: %s)", target, target_source)
    LOG.info("Target absolute: %s", target_abs)
    LOG.info("$ORIGIN:         %s", origin)

    needed, rpaths = read_elf_deps(target)

    search_dirs: list[Path] = []
    for rp in rpaths:
        if args.expand_origin:
            search_dirs.append(expand_origin(rp, origin))
        else:
            search_dirs.append(Path(rp))
    search_dirs.append(lib_dir)
    if runtime_sys.exists():
        search_dirs.append(runtime_sys)
    runtime_nvidia = sdk_root / "runtime" / "nvidia"
    if runtime_nvidia.exists():
        for d in runtime_nvidia.glob("*/lib"):
            search_dirs.append(d)

    if args.check_host:
        for d in [
            Path("/lib"),
            Path("/lib64"),
            Path("/usr/lib"),
            Path("/usr/lib64"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]:
            search_dirs.append(d)

    search_dirs = unique_dirs(search_dirs)

    missing_any = False
    deps_json: list[dict[str, object]] = []

    for name in needed:
        if name in SYSTEM_ALLOWLIST and args.allow_host:
            LOG.debug("[OK host] %s", name)
            deps_json.append({"name": name, "status": "ok", "classification": "host"})
            continue

        found: FoundDep | None = None
        for d in search_dirs:
            cand = d / name
            if cand.exists():
                found = FoundDep(name, d, cand)
                break

        if not found:
            LOG.warning("[MISSING] %s", name)
            deps_json.append({"name": name, "status": "missing"})
            if not args.allow_host:
                missing_any = True
                if args.fail_fast:
                    die(f"Unresolved dependency: {name}")
            continue

        cls = classify(found.full_path, sdk_root)
        shown = display_path(
            found,
            sdk_root=sdk_root,
            origin=origin,
            expand_origin=args.expand_origin,
            resolve_symlinks=args.resolve_symlinks,
        )

        LOG.debug("[%s] %s -> %s", cls, name, shown)

        deps_json.append(
            {
                "name": name,
                "status": "ok",
                "classification": cls,
                "path": shown,
                "absolute_path": str(found.full_path.resolve()),
            }
        )

    llvm = len(list(lib_dir.glob("libLLVM*.so*")))
    mlir = len(list(lib_dir.glob("libMLIR*.so*")))

    if args.json:
        json.dump(
            {
                "metadata": {
                    "context": {
                        "sdk_root": str(sdk_root),
                        "sdk_root_source": sdk_root_source,
                        "target": str(target),
                        "target_absolute": str(target_abs),
                        "origin": str(origin),
                    },
                },
                "dependencies": deps_json,
                "counts": {"llvm_dsos": llvm, "mlir_dsos": mlir},
                "status": "fail" if missing_any else "pass",
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        LOG.info("LLVM DSOs: %d", llvm)
        LOG.info("MLIR DSOs: %d", mlir)
        if missing_any:
            die("One or more dependencies could not be resolved")
        LOG.info("PASS: all dependencies resolved")

    if missing_any:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
