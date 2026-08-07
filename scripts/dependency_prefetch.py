"""Complete dependency proposals with hashes from their Nix fetchers."""

# pyright: reportImplicitRelativeImport=false

from __future__ import annotations

import copy
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from dependency_proposal import ChangeRecord, DependencyProposal, UnresolvedRecord
from dependency_schema import DependencySnapshot, SnapshotEntry, SourceKind


@dataclass(frozen=True)
class CommandOutput:
    stdout: str
    stderr: str


type CommandRunner = Callable[[Sequence[str]], CommandOutput]


def _required(entry: SnapshotEntry, field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str):
        message = f"source entry lacks required field {field!r}"
        raise ValueError(message)
    return value


def run_command(arguments: Sequence[str]) -> CommandOutput:
    """Run one prefetch command and capture its machine-readable result."""
    try:
        result = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        command = arguments[0] if arguments else "prefetch command"
        message = f"{command} failed while completing the dependency proposal"
        raise RuntimeError(message) from error
    return CommandOutput(result.stdout, result.stderr)


def _prefetch_file(url: str, *, unpack: bool, run: CommandRunner) -> str:
    arguments = ["nix", "store", "prefetch-file", "--json", "--no-pretty"]
    if unpack:
        arguments.append("--unpack")
    arguments.append(url)
    output = run(arguments)
    parsed = cast("object", json.loads(output.stdout))
    if not isinstance(parsed, dict):
        message = "nix store prefetch-file returned a non-object result"
        raise TypeError(message)
    mapping = cast("dict[object, object]", parsed)
    hash_value = mapping.get("hash")
    if not isinstance(hash_value, str):
        message = "nix store prefetch-file result lacks a hash"
        raise TypeError(message)
    return hash_value


def _prefetch_git(entry: SnapshotEntry, run: CommandRunner) -> str:
    owner = _required(entry, "owner")
    repository = _required(entry, "repo")
    arguments = [
        "nix-prefetch-git",
        "--fetch-submodules",
        "--url",
        f"https://github.com/{owner}/{repository}.git",
        "--rev",
        entry["rev"],
    ]
    output = run(arguments)
    match = re.search(r"^hash is (\S+)$", output.stderr, re.MULTILINE)
    if match is None:
        message = "nix-prefetch-git result lacks a hash"
        raise ValueError(message)
    converted = run(
        [
            "nix",
            "hash",
            "convert",
            "--hash-algo",
            "sha256",
            "--to",
            "sri",
            match.group(1),
        ],
    )
    return converted.stdout.strip()


def prefetch_source(entry: SnapshotEntry, run: CommandRunner = run_command) -> str:
    """Compute the hash produced by the source entry's declared Nix fetcher."""
    if entry["type"] == SourceKind.url:
        return _prefetch_file(_required(entry, "url"), unpack=False, run=run)
    if entry.get("fetch_submodules", False):
        return _prefetch_git(entry, run)
    owner = _required(entry, "owner")
    repository = _required(entry, "repo")
    archive = f"https://github.com/{owner}/{repository}/archive/{entry['rev']}.tar.gz"
    return _prefetch_file(archive, unpack=True, run=run)


def _apply_change(snapshot: DependencySnapshot, change: ChangeRecord) -> None:
    proposed = change["proposed"]
    if proposed is None:
        return
    entry = cast(
        "dict[str, object]",
        cast("object", snapshot[change["source"]]),
    )
    entry[change["field"]] = proposed


def complete_proposal(
    snapshot: DependencySnapshot,
    proposal: DependencyProposal,
    run: CommandRunner = run_command,
) -> DependencyProposal:
    """Prefetch unresolved sources and attach a complete proposed snapshot."""
    completed = copy.deepcopy(proposal)
    proposed_snapshot = copy.deepcopy(snapshot)
    for change in completed["changes"]:
        _apply_change(proposed_snapshot, change)

    for change in completed["changes"]:
        if change["field"] != "hash" or change["proposed"] is not None:
            continue
        source = change["source"]
        fetched_hash = prefetch_source(proposed_snapshot[source], run)
        change["proposed"] = fetched_hash
        change["authority"] = "Nix fetcher prefetch"
        _apply_change(proposed_snapshot, change)

    unresolved: list[UnresolvedRecord] = [
        {
            "source": change["source"],
            "field": change["field"],
            "reason": change["authority"],
        }
        for change in completed["changes"]
        if change["proposed"] is None
    ]
    completed["complete"] = not unresolved
    completed["unresolved"] = unresolved
    completed["snapshot"] = proposed_snapshot
    return completed
