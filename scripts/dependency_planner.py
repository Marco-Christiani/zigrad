"""Resolve LLVM requirements for a Zigrad build configuration."""

# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from dependency_metadata import XlaMetadata, llvm_version
from dependency_schema import BuildManifest, CompatibilityEntry, DependencySnapshot

Version = tuple[int, int, int]


class CandidateRecord(TypedDict):
    name: str
    revision: str
    version: str
    selection: str


class RequirementRecord(TypedDict):
    consumer: str
    kind: str
    authority: str
    revision: str | None
    minimum_version: str | None
    satisfied: bool
    explanation: str


class GroupPlan(TypedDict):
    name: str
    isolation: list[str]
    candidate: CandidateRecord
    requirements: list[RequirementRecord]
    compatible: bool
    verification: str


class DependencyPlan(TypedDict):
    configuration: str
    package: str
    groups: list[GroupPlan]
    compatible: bool


@dataclass(frozen=True)
class Requirement:
    consumer: str
    kind: str
    authority: str
    revision: str | None = None
    minimum_version: Version | None = None


@dataclass(frozen=True)
class Candidate:
    name: Literal["iree_llvm", "llvm"]
    revision: str
    version: Version
    selection: str


@dataclass(frozen=True)
class RequirementResult:
    requirement: Requirement
    satisfied: bool
    explanation: str


def format_version(version: Version) -> str:
    return ".".join(str(component) for component in version)


def tvm_minimum_llvm(source: Path) -> Version:
    path = source / "cmake/modules/LLVM.cmake"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"TVM_LLVM_VERSION\}\s+LESS\s+([0-9]+)", text)
    if not match:
        raise ValueError(f"{path} lacks TVM's minimum LLVM check")

    encoded = int(match.group(1))
    return encoded // 10, encoded % 10, 0


def requirement_for(
    name: str,
    consumer: str,
    snapshot: DependencySnapshot,
    xla_source: Path,
    tvm_source: Path,
    iree_llvm_revision: str | None,
) -> Requirement:
    if name == "xla-llvm":
        return Requirement(
            consumer=consumer,
            kind="exact-revision",
            revision=XlaMetadata(xla_source).llvm_revision(),
            authority="XLA third_party/llvm/workspace.bzl",
        )
    if name == "tvm-llvm":
        return Requirement(
            consumer=consumer,
            kind="minimum-version",
            minimum_version=tvm_minimum_llvm(tvm_source),
            authority="TVM cmake/modules/LLVM.cmake",
        )
    if name == "iree-llvm":
        return Requirement(
            consumer=consumer,
            kind="exact-revision",
            revision=iree_llvm_revision or snapshot["iree_llvm"]["rev"],
            authority="IREE gitlink checked by check-dependency-snapshot",
        )
    raise ValueError(f"unknown LLVM requirement adapter {name!r}")


def select_candidate(
    group: str,
    requirements: list[Requirement],
    snapshot: DependencySnapshot,
    llvm_source: Path,
    iree_llvm_source: Path,
) -> Candidate:
    source_name: Literal["iree_llvm", "llvm"] = (
        "iree_llvm" if group == "iree-build-llvm" else "llvm"
    )
    source = iree_llvm_source if source_name == "iree_llvm" else llvm_source
    selected = snapshot[source_name]
    exact_revisions = {
        requirement.revision
        for requirement in requirements
        if requirement.kind == "exact-revision"
    }
    if len(exact_revisions) > 1:
        revisions = ", ".join(sorted(revision for revision in exact_revisions if revision))
        raise ValueError(f"{group} has conflicting exact LLVM revisions: {revisions}")

    if exact_revisions:
        required_revision = next(iter(exact_revisions))
        selection = "exact consumer requirement"
        revision = required_revision or ""
    else:
        selection = "Zigrad dependency snapshot policy"
        revision = selected["rev"]

    return Candidate(
        name=source_name,
        revision=revision,
        version=llvm_version(source),
        selection=selection,
    )


def evaluate_requirement(
    requirement: Requirement,
    candidate: Candidate,
    available_revision: str,
) -> RequirementResult:
    if candidate.revision != available_revision:
        return RequirementResult(
            requirement=requirement,
            satisfied=False,
            explanation=(
                f"selected revision {candidate.revision} is not materialized by the "
                f"snapshot candidate {available_revision}"
            ),
        )
    if requirement.kind == "exact-revision":
        satisfied = candidate.revision == requirement.revision
        return RequirementResult(
            requirement=requirement,
            satisfied=satisfied,
            explanation=(
                f"requires revision {requirement.revision}"
                if satisfied
                else f"requires revision {requirement.revision}, selected {candidate.revision}"
            ),
        )
    if requirement.kind == "minimum-version":
        minimum = requirement.minimum_version or (0, 0, 0)
        satisfied = candidate.version >= minimum
        return RequirementResult(
            requirement=requirement,
            satisfied=satisfied,
            explanation=(
                f"requires LLVM >= {format_version(minimum)}, selected "
                f"{format_version(candidate.version)}"
            ),
        )
    raise ValueError(f"unknown requirement kind {requirement.kind!r}")


def plan_configuration(
    name: str,
    manifest: BuildManifest,
    snapshot: DependencySnapshot,
    xla_source: Path,
    llvm_source: Path,
    tvm_source: Path,
    iree_llvm_source: Path,
    iree_llvm_revision: str | None = None,
) -> DependencyPlan:
    if name not in manifest:
        available = ", ".join(sorted(manifest))
        raise ValueError(f"unknown configuration {name!r}. Available: {available}")

    configuration = manifest[name]
    entries = configuration["compatibility"]
    grouped: dict[str, list[CompatibilityEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry["group"], []).append(entry)

    groups: list[GroupPlan] = []
    for group_name, group_entries in sorted(grouped.items()):
        requirements = [
            requirement_for(
                entry["requirement"],
                entry["consumer"],
                snapshot,
                xla_source,
                tvm_source,
                iree_llvm_revision,
            )
            for entry in group_entries
        ]
        candidate = select_candidate(
            group_name,
            requirements,
            snapshot,
            llvm_source,
            iree_llvm_source,
        )
        available_revision = snapshot[candidate.name]["rev"]
        results = [
            evaluate_requirement(requirement, candidate, available_revision)
            for requirement in requirements
        ]
        groups.append(
            {
                "name": group_name,
                "isolation": sorted({entry["isolation"] for entry in group_entries}),
                "candidate": {
                    "name": candidate.name,
                    "revision": candidate.revision,
                    "version": format_version(candidate.version),
                    "selection": candidate.selection,
                },
                "requirements": [
                    {
                        "consumer": result.requirement.consumer,
                        "kind": result.requirement.kind,
                        "authority": result.requirement.authority,
                        "revision": result.requirement.revision,
                        "minimum_version": (
                            format_version(result.requirement.minimum_version)
                            if result.requirement.minimum_version is not None
                            else None
                        ),
                        "satisfied": result.satisfied,
                        "explanation": result.explanation,
                    }
                    for result in results
                ],
                "compatible": all(result.satisfied for result in results),
                "verification": "not evaluated by the planner",
            },
        )

    return {
        "configuration": name,
        "package": configuration["packageName"],
        "groups": groups,
        "compatible": all(group["compatible"] for group in groups),
    }
