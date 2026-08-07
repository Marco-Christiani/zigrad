"""Validate relationships within the locked external dependency snapshot."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from dependency_metadata import (
    COMPONENT_SPECS,
    XlaCudaMetadata,
    llvm_version,
    quoted_assignment,
)


class CheckFailure(Exception):
    pass


def source_revision(snapshot: dict[str, Any], source_name: str) -> str:
    try:
        return snapshot[source_name]["rev"]
    except KeyError as error:
        raise CheckFailure(f"external-sources.json lacks a revision for {source_name}") from error


def cuda_selection(flake_path: Path) -> dict[str, str]:
    text = flake_path.read_text(encoding="utf-8")
    fields = ("cudaVersion", "cudnnVersion", "ncclVersion", "nvshmemVersion")
    selection: dict[str, str] = {}
    for field in fields:
        match = re.search(
            rf'^\s*{field}\s*=\s*"([^"]+)";',
            text,
            re.MULTILINE,
        )
        if not match:
            raise CheckFailure(f"{flake_path} lacks cudaCfg.{field}")
        selection[field] = match.group(1)
    return selection


def check_xla(repo: Path, source: Path, snapshot: dict[str, Any]) -> None:
    relationships = (
        (
            source / "third_party/llvm/workspace.bzl",
            "LLVM_COMMIT",
            "llvm",
        ),
        (
            source / "third_party/stablehlo/workspace.bzl",
            "STABLEHLO_COMMIT",
            "stablehlo",
        ),
    )
    for path, variable, input_name in relationships:
        try:
            declared = quoted_assignment(path, variable)
        except ValueError as error:
            raise CheckFailure(str(error)) from error
        selected = source_revision(snapshot, input_name)
        if declared != selected:
            raise CheckFailure(
                f"{input_name} is {selected}, but XLA declares {declared}",
            )

    selection = cuda_selection(repo / "flake.nix")
    cuda_version = selection["cudaVersion"]
    defaults = XlaCudaMetadata(source).defaults("pjrt_cuda12")
    if cuda_version != defaults.cuda:
        raise CheckFailure(
            f"cudaCfg selects {cuda_version}, but XLA pjrt_cuda12 selects {defaults.cuda}",
        )

    catalog_path = repo / "nix/cuda-redist.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    try:
        entry = catalog["cuda"][cuda_version]
    except KeyError as error:
        raise CheckFailure(f"{catalog_path} lacks CUDA {cuda_version}") from error

    expected_versions = {
        "cudnn_version": selection["cudnnVersion"],
        "nccl_version": selection["ncclVersion"],
        "nvshmem_version": selection["nvshmemVersion"],
    }
    for field, expected in expected_versions.items():
        actual = entry.get(field)
        if actual != expected:
            raise CheckFailure(
                f"CUDA {cuda_version} records {field}={actual}, cudaCfg selects {expected}",
            )

    integration_versions = {
        "cuDNN": (selection["cudnnVersion"], defaults.cudnn),
        "NVSHMEM": (selection["nvshmemVersion"], defaults.nvshmem),
    }
    for name, (selected, required_version) in integration_versions.items():
        if selected != required_version:
            raise CheckFailure(
                f"cudaCfg selects {name} {selected}, but XLA pjrt_cuda12 requires {required_version}",
            )

    required = {spec.manifest_key for spec in COMPONENT_SPECS} | {"nccl"}
    missing = required - entry.get("components", {}).keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise CheckFailure(f"CUDA {cuda_version} lacks required components: {names}")


def github_tree(owner: str, repository: str, revision: str) -> list[dict[str, Any]]:
    url = (
        f"https://api.github.com/repos/{owner}/{repository}/git/trees/"
        f"{revision}?recursive=1"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "zigrad-dependency-snapshot/1.0",
    }
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60.0) as response:
        result = json.loads(response.read())
    if result.get("truncated"):
        raise CheckFailure(f"GitHub returned a truncated tree for {owner}/{repository}")
    return result["tree"]


def check_iree(snapshot: dict[str, Any]) -> None:
    revision = source_revision(snapshot, "iree")
    tree = github_tree("iree-org", "iree", revision)
    gitlinks = {
        entry["path"]: entry["sha"]
        for entry in tree
        if entry.get("type") == "commit"
    }
    relationships = {
        "third_party/llvm-project": "iree_llvm",
        "third_party/stablehlo": "iree_stablehlo",
        "third_party/flatcc": "iree_flatcc",
        "third_party/benchmark": "iree_benchmark",
    }
    for path, input_name in relationships.items():
        declared = gitlinks.get(path)
        if declared is None:
            raise CheckFailure(f"IREE {revision} lacks gitlink {path}")

        selected = source_revision(snapshot, input_name)
        if declared != selected:
            raise CheckFailure(
                f"{input_name} is {selected}, but IREE declares {declared}",
            )


def zon_dependency(text: str, name: str) -> tuple[str, str]:
    block = re.search(
        rf"\.{re.escape(name)}\s*=\s*\.\{{(.*?)\n\s*\}},",
        text,
        re.DOTALL,
    )
    if not block:
        raise CheckFailure(f"build.zig.zon lacks dependency {name}")

    url = re.search(r'\.url\s*=\s*"([^"]+)"', block.group(1))
    package_hash = re.search(r'\.hash\s*=\s*"([^"]+)"', block.group(1))
    if not url or not package_hash:
        raise CheckFailure(f"build.zig.zon has incomplete dependency {name}")

    revision = url.group(1).rsplit("#", 1)[-1]
    return revision, package_hash.group(1)


def nix_zig_dependency(text: str, name: str) -> tuple[str, str]:
    block = re.search(
        rf"\b{re.escape(name)}\s*=\s*\{{(.*?)\n\s*\}};",
        text,
        re.DOTALL,
    )
    if not block:
        raise CheckFailure(f"zig-dependencies.nix lacks dependency {name}")

    revision = re.search(r'\brev\s*=\s*"([^"]+)"', block.group(1))
    package_name = re.search(r'\bname\s*=\s*"([^"]+)"', block.group(1))
    if not revision or not package_name:
        raise CheckFailure(f"zig-dependencies.nix has incomplete dependency {name}")
    return revision.group(1), package_name.group(1)


def check_zig_dependencies(repo: Path) -> None:
    zon = (repo / "build.zig.zon").read_text(encoding="utf-8")
    nix = (repo / "nix/packages/zig-dependencies.nix").read_text(encoding="utf-8")
    names = {
        "safetensors_zg": "safetensors",
        "protobuf": "protobuf",
    }
    for zon_name, nix_name in names.items():
        expected = zon_dependency(zon, zon_name)
        actual = nix_zig_dependency(nix, nix_name)
        if actual != expected:
            raise CheckFailure(
                f"Nix metadata for {zon_name} is {actual}, build.zig.zon records {expected}",
            )


def check_llvm(snapshot: dict[str, Any], source: Path) -> None:
    selected = snapshot["llvm"].get("version")
    actual = ".".join(str(component) for component in llvm_version(source))
    if selected != actual:
        raise CheckFailure(
            f"external-sources.json records LLVM {selected}, source reports {actual}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the locked Zigrad external dependency snapshot.",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--xla-src", type=Path, required=True)
    parser.add_argument("--llvm-src", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the IREE gitlink check, which queries the exact GitHub tree",
    )
    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    snapshot = json.loads(
        (repo / "nix/external-sources.json").read_text(encoding="utf-8"),
    )
    check_xla(repo, arguments.xla_src, snapshot)
    check_llvm(snapshot, arguments.llvm_src)
    if not arguments.offline:
        check_iree(snapshot)
    check_zig_dependencies(repo)
    print("dependency snapshot is consistent")


if __name__ == "__main__":
    try:
        main()
    except CheckFailure as error:
        raise SystemExit(f"dependency snapshot check failed: {error}") from error
