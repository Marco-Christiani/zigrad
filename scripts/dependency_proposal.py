"""Derive dependency snapshot changes from local integration candidates."""

# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from dependency_metadata import XlaMetadata
from dependency_schema import (
    BuildConfiguration,
    BuildManifest,
    DependencySnapshot,
    IntegrationRoot,
)


@dataclass(frozen=True)
class CandidateSource:
    name: IntegrationRoot
    path: Path
    revision: str


class CandidateRecord(TypedDict):
    path: str
    revision: str


class ChangeRecord(TypedDict):
    source: str
    field: str
    current: str | None
    proposed: str | None
    changed: bool
    authority: str


class UnresolvedRecord(TypedDict):
    source: str
    field: str
    reason: str


type ConstraintValue = str | dict[str, "ConstraintValue"]
type SnapshotField = Literal["hash", "rev", "source_root", "url"]


class DependencyProposal(TypedDict):
    configuration: str
    demanded_roots: list[IntegrationRoot]
    candidates: dict[IntegrationRoot, CandidateRecord]
    changes: list[ChangeRecord]
    constraints: dict[str, dict[str, ConstraintValue]]
    complete: bool
    unresolved: list[UnresolvedRecord]
    verification: str
    snapshot: NotRequired[DependencySnapshot]


ROOT_NODES: dict[IntegrationRoot, frozenset[str]] = {
    IntegrationRoot.xla: frozenset(
        {
            "pjrt-api",
            "pjrt-cpu",
            "pjrt-cuda",
            "stablehlo-mlir",
        },
    ),
    IntegrationRoot.iree: frozenset({"iree"}),
    IntegrationRoot.tvm: frozenset({"tvm-cpu", "tvm-cuda", "tvm-python-cuda"}),
    IntegrationRoot.mirage: frozenset({"mirage-cuda"}),
}

IREE_GITLINKS = {
    "iree_benchmark": "third_party/benchmark",
    "iree_flatcc": "third_party/flatcc",
    "iree_llvm": "third_party/llvm-project",
    "iree_stablehlo": "third_party/stablehlo",
}


def git_revision(path: Path) -> str:
    return resolve_git_revision(path, "HEAD")


def resolve_git_revision(path: Path, revision: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", revision],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot resolve {revision!r} in candidate {path}") from error


def candidate_source(
    name: IntegrationRoot,
    path: Path,
    snapshot_revision: str,
) -> CandidateSource:
    head = git_revision(path)
    try:
        selected = resolve_git_revision(path, snapshot_revision)
    except ValueError:
        selected = None
    revision = snapshot_revision if selected == head else head
    return CandidateSource(name, path, revision)


def gitlink_revision(path: Path, relative_path: str) -> str:
    try:
        output = subprocess.run(
            ["git", "-C", str(path), "ls-tree", "HEAD", relative_path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"failed to read {relative_path} from {path}") from error

    fields = output.split()
    if len(fields) < 3 or fields[1] != "commit":
        raise ValueError(f"{path} does not record the {relative_path} gitlink")
    return fields[2]


def demanded_roots(configuration: BuildConfiguration) -> set[IntegrationRoot]:
    resolved = set(configuration["resolved"])
    return {
        name for name, nodes in ROOT_NODES.items() if not resolved.isdisjoint(nodes)
    }


def changed_field(
    snapshot: DependencySnapshot,
    source: str,
    field: SnapshotField,
    proposed: str,
    authority: str,
) -> ChangeRecord:
    current = snapshot[source].get(field)
    return {
        "source": source,
        "field": field,
        "current": current,
        "proposed": proposed,
        "changed": current != proposed,
        "authority": authority,
    }


def candidate_root_changes(
    candidate: CandidateSource,
    snapshot: DependencySnapshot,
) -> list[ChangeRecord]:
    changes = [
        changed_field(
            snapshot,
            candidate.name,
            "rev",
            candidate.revision,
            f"Git HEAD of {candidate.path}",
        ),
    ]
    if candidate.revision != snapshot[candidate.name]["rev"]:
        changes.append(
            {
                "source": candidate.name,
                "field": "hash",
                "current": snapshot[candidate.name].get("hash"),
                "proposed": None,
                "changed": True,
                "authority": "requires source prefetch",
            },
        )
    return changes


def xla_changes(
    candidate: CandidateSource,
    snapshot: DependencySnapshot,
) -> tuple[list[ChangeRecord], dict[str, ConstraintValue]]:
    metadata = XlaMetadata(candidate.path)
    llvm = metadata.llvm_requirement()
    stablehlo = metadata.stablehlo_requirement()
    cuda = metadata.defaults("pjrt_cuda12")
    changes = candidate_root_changes(candidate, snapshot)

    for source, requirement, authority in (
        ("llvm", llvm, "XLA third_party/llvm/workspace.bzl"),
        (
            "stablehlo",
            stablehlo,
            "XLA third_party/stablehlo/workspace.bzl",
        ),
    ):
        revision_change = changed_field(
            snapshot,
            source,
            "rev",
            requirement.revision,
            authority,
        )
        changes.append(revision_change)
        if revision_change["changed"]:
            changes.append(
                {
                    "source": source,
                    "field": "hash",
                    "current": snapshot[source].get("hash"),
                    "proposed": None,
                    "changed": True,
                    "authority": "requires source prefetch",
                },
            )

    return changes, {
        "cuda": {
            "cuda": cuda.cuda,
            "cudnn": cuda.cudnn,
            "nvshmem": cuda.nvshmem,
        },
        "companions": {
            "llvm": {
                "revision": llvm.revision,
                "upstream_archive_hash_sri": llvm.archive_hash_sri,
            },
            "stablehlo": {
                "revision": stablehlo.revision,
                "upstream_archive_hash_sri": stablehlo.archive_hash_sri,
            },
        },
    }


def iree_changes(
    candidate: CandidateSource,
    snapshot: DependencySnapshot,
) -> list[ChangeRecord]:
    changes = candidate_root_changes(candidate, snapshot)
    for source, relative_path in IREE_GITLINKS.items():
        revision = gitlink_revision(candidate.path, relative_path)
        changes.append(
            changed_field(
                snapshot,
                source,
                "rev",
                revision,
                f"IREE gitlink {relative_path}",
            ),
        )
        if revision != snapshot[source]["rev"]:
            changes.append(
                {
                    "source": source,
                    "field": "hash",
                    "current": snapshot[source].get("hash"),
                    "proposed": None,
                    "changed": True,
                    "authority": "requires source prefetch",
                },
            )
    return changes


def mirage_changes(
    candidate: CandidateSource,
    snapshot: DependencySnapshot,
) -> list[ChangeRecord]:
    changes = candidate_root_changes(candidate, snapshot)
    entry = snapshot[IntegrationRoot.mirage]
    current_revision = entry["rev"]
    revision_fields: tuple[
        tuple[Literal["url", "source_root"], str | None],
        ...,
    ] = (
        ("url", entry.get("url")),
        ("source_root", entry.get("source_root")),
    )
    for field, current in revision_fields:
        if current is None:
            continue
        if current_revision not in current:
            message = f"Mirage {field} does not contain its selected revision"
            raise ValueError(message)
        changes.append(
            changed_field(
                snapshot,
                IntegrationRoot.mirage,
                field,
                current.replace(current_revision, candidate.revision),
                f"Mirage {field} revision substitution",
            ),
        )
    return changes


def propose_configuration(
    name: str,
    manifest: BuildManifest,
    snapshot: DependencySnapshot,
    candidates: dict[IntegrationRoot, CandidateSource],
) -> DependencyProposal:
    if name not in manifest:
        available = ", ".join(sorted(manifest))
        raise ValueError(f"unknown configuration {name!r}. Available: {available}")

    roots = demanded_roots(manifest[name])
    unsupported = set(candidates) - roots
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"configuration {name!r} does not demand candidate roots: {names}"
        )

    changes: list[ChangeRecord] = []
    constraints: dict[str, dict[str, ConstraintValue]] = {}
    for candidate_name, candidate in sorted(candidates.items()):
        if candidate_name == IntegrationRoot.xla:
            xla_source_changes, xla_constraints = xla_changes(candidate, snapshot)
            changes.extend(xla_source_changes)
            constraints["xla"] = xla_constraints
        elif candidate_name == IntegrationRoot.iree:
            changes.extend(iree_changes(candidate, snapshot))
        elif candidate_name == IntegrationRoot.mirage:
            changes.extend(mirage_changes(candidate, snapshot))
        else:
            changes.extend(candidate_root_changes(candidate, snapshot))

    unresolved: list[UnresolvedRecord] = [
        {
            "source": change["source"],
            "field": change["field"],
            "reason": change["authority"],
        }
        for change in changes
        if change["changed"] and change["proposed"] is None
    ]
    return {
        "configuration": name,
        "demanded_roots": sorted(roots),
        "candidates": {
            candidate.name: {
                "path": str(candidate.path),
                "revision": candidate.revision,
            }
            for candidate in candidates.values()
        },
        "changes": [change for change in changes if change["changed"]],
        "constraints": constraints,
        "complete": not unresolved,
        "unresolved": unresolved,
        "verification": "not evaluated by the proposal",
    }
