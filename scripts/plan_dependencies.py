"""Explain dependency selection for a Zigrad build configuration."""

# ruff: noqa: INP001
# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Protocol, cast

from dependency_planner import DependencyPlan, plan_configuration
from dependency_proposal import (
    ConstraintValue,
    DependencyProposal,
    candidate_source,
    propose_configuration,
)
from dependency_schema import IntegrationRoot, read_manifest, read_snapshot


class _Arguments(Protocol):
    configuration: str
    manifest: Path
    snapshot: Path
    xla_src: Path
    llvm_src: Path
    tvm_src: Path
    iree_llvm_src: Path
    candidate: list[tuple[IntegrationRoot, Path]]
    json: bool


def _write_line(line: str = "") -> None:
    _ = sys.stdout.write(f"{line}\n")


def _print_plan(plan: DependencyPlan) -> None:
    _write_line(f"configuration: {plan['configuration']} ({plan['package']})")
    if not plan["groups"]:
        _write_line("LLVM groups: none")
        return

    for group in plan["groups"]:
        candidate = group["candidate"]
        isolation = ", ".join(group["isolation"])
        status = "compatible" if group["compatible"] else "incompatible"
        _write_line()
        _write_line(f"{group['name']} [{isolation}]: {status}")
        selected = (
            f"  selected {candidate['name']}@{candidate['revision']}"
            f" (LLVM {candidate['version']})"
        )
        _write_line(selected)
        _write_line(f"  selection: {candidate['selection']}")
        for requirement in group["requirements"]:
            marker = "ok" if requirement["satisfied"] else "fail"
            summary = f"  [{marker}] {requirement['consumer']}: {requirement['explanation']}"
            _write_line(summary)
            _write_line(f"       authority: {requirement['authority']}")
        _write_line("  verification: not evaluated by the planner")


def _parse_candidate(value: str) -> tuple[IntegrationRoot, Path]:
    try:
        name, path_text = value.split("=", 1)
    except ValueError as error:
        message = "expected NAME=PATH"
        raise argparse.ArgumentTypeError(message) from error
    if not name or not path_text:
        message = "expected NAME=PATH"
        raise argparse.ArgumentTypeError(message)
    try:
        root = IntegrationRoot(name)
    except ValueError as error:
        message = f"unsupported candidate root {name!r}"
        raise argparse.ArgumentTypeError(message) from error
    path = Path(path_text).resolve()
    if not path.is_dir():
        message = f"candidate path is not a directory: {path}"
        raise argparse.ArgumentTypeError(message)
    return root, path


def _print_proposal(proposal: DependencyProposal) -> None:
    _write_line()
    _write_line("candidate proposal:")
    if not proposal["candidates"]:
        _write_line("  candidates: none")
        return

    for root, candidate in sorted(proposal["candidates"].items()):
        _write_line(f"  {root}@{candidate['revision']} from {candidate['path']}")
    if not proposal["changes"]:
        _write_line("  snapshot changes: none")
    else:
        _write_line("  snapshot changes:")
        for change in proposal["changes"]:
            proposed = change["proposed"] or "unresolved"
            summary = (
                f"    {change['source']}.{change['field']}:"
                f" {change['current']} -> {proposed}"
            )
            _write_line(summary)
            _write_line(f"      authority: {change['authority']}")
    if proposal["constraints"]:
        _write_line("  derived constraints:")
        for constraint_name, values in sorted(proposal["constraints"].items()):
            _write_line(f"    {constraint_name}:")
            _print_mapping(values, 6)
    status = "complete" if proposal["complete"] else "requires prefetch"
    _write_line(f"  proposal status: {status}")
    _write_line("  verification: not evaluated by the proposal")


def _print_mapping(values: dict[str, ConstraintValue], indentation: int) -> None:
    prefix = " " * indentation
    for key, value in sorted(values.items()):
        if isinstance(value, dict):
            _write_line(f"{prefix}{key}:")
            _print_mapping(value, indentation + 2)
        else:
            _write_line(f"{prefix}{key}: {value}")


def main() -> None:
    """Run the dependency planner CLI."""
    parser = argparse.ArgumentParser(
        description="Explain LLVM compatibility for a Zigrad configuration.",
    )
    _ = parser.add_argument("--configuration", required=True)
    _ = parser.add_argument("--manifest", type=Path, required=True)
    _ = parser.add_argument("--snapshot", type=Path, required=True)
    _ = parser.add_argument("--xla-src", type=Path, required=True)
    _ = parser.add_argument("--llvm-src", type=Path, required=True)
    _ = parser.add_argument("--tvm-src", type=Path, required=True)
    _ = parser.add_argument("--iree-llvm-src", type=Path, required=True)
    _ = parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        type=_parse_candidate,
        metavar="NAME=PATH",
        help="derive a proposal from a local XLA, IREE, TVM, or Mirage worktree",
    )
    _ = parser.add_argument("--json", action="store_true")
    arguments = cast("_Arguments", cast("object", parser.parse_args()))

    source_paths = {
        IntegrationRoot.xla: arguments.xla_src,
        IntegrationRoot.tvm: arguments.tvm_src,
    }
    candidate_paths = dict(arguments.candidate)
    source_paths.update(candidate_paths)

    manifest = read_manifest(arguments.manifest)
    snapshot = read_snapshot(arguments.snapshot)
    candidates = {
        name: candidate_source(
            name,
            path,
            snapshot[name]["rev"],
        )
        for name, path in candidate_paths.items()
    }
    proposal = propose_configuration(
        arguments.configuration,
        manifest,
        snapshot,
        candidates,
    )
    proposed_iree_llvm = next(
        (
            change["proposed"]
            for change in proposal["changes"]
            if change["source"] == "iree_llvm" and change["field"] == "rev"
        ),
        None,
    )
    plan = plan_configuration(
        arguments.configuration,
        manifest,
        snapshot,
        source_paths[IntegrationRoot.xla],
        arguments.llvm_src,
        source_paths[IntegrationRoot.tvm],
        arguments.iree_llvm_src,
        proposed_iree_llvm,
    )
    if arguments.json:
        _write_line(
            json.dumps(
                {"plan": plan, "proposal": proposal}, indent=2, sort_keys=True,
            ),
        )
    else:
        _print_plan(plan)
        _print_proposal(proposal)

    if not plan["compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
