"""Parse the dependency planner's committed JSON inputs."""

from __future__ import annotations

import json
from enum import StrEnum, auto
from pathlib import Path
from typing import NotRequired, TypedDict, cast


class IntegrationRoot(StrEnum):
    iree = auto()
    mirage = auto()
    tvm = auto()
    xla = auto()


class SourceKind(StrEnum):
    github = auto()
    url = auto()


class RequirementAdapter(StrEnum):
    iree_llvm = "iree-llvm"
    tvm_llvm = "tvm-llvm"
    xla_llvm = "xla-llvm"


class Isolation(StrEnum):
    in_process = "in-process"
    out_of_process = "out-of-process"


class SnapshotEntry(TypedDict):
    type: SourceKind
    rev: str
    hash: str
    owner: NotRequired[str]
    repo: NotRequired[str]
    url: NotRequired[str]
    source_root: NotRequired[str]
    fetch_submodules: NotRequired[bool]
    version: NotRequired[str]


type DependencySnapshot = dict[str, SnapshotEntry]


class CompatibilityEntry(TypedDict):
    consumer: str
    group: str
    isolation: Isolation
    requirement: RequirementAdapter


class BuildConfiguration(TypedDict):
    packageName: str
    resolved: list[str]
    compatibility: list[CompatibilityEntry]


type BuildManifest = dict[str, BuildConfiguration]


def _read_object(path: Path) -> dict[str, object]:
    parsed = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(parsed, dict):
        message = f"expected a JSON object in {path}"
        raise TypeError(message)

    mapping = cast("dict[object, object]", parsed)
    if not all(isinstance(key, str) for key in mapping):
        message = f"expected string keys in JSON object {path}"
        raise TypeError(message)
    return {cast("str", key): value for key, value in mapping.items()}


def _read_string(mapping: dict[str, object], field: str, context: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str):
        message = f"expected string {field!r} in {context}"
        raise TypeError(message)
    return value


def _read_enum[EnumT: StrEnum](
    enum_type: type[EnumT],
    mapping: dict[str, object],
    field: str,
    context: str,
) -> EnumT:
    value = _read_string(mapping, field, context)
    try:
        return enum_type(value)
    except ValueError as error:
        message = f"unsupported {field} {value!r} in {context}"
        raise ValueError(message) from error


def _read_string_list(
    mapping: dict[str, object],
    field: str,
    context: str,
) -> list[str]:
    value = mapping.get(field)
    if not isinstance(value, list):
        message = f"expected string list {field!r} in {context}"
        raise TypeError(message)
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        message = f"expected string list {field!r} in {context}"
        raise TypeError(message)
    return [cast("str", item) for item in items]


def read_snapshot(path: Path) -> DependencySnapshot:
    """Read source revisions and optional materialization metadata."""
    raw = _read_object(path)
    snapshot: DependencySnapshot = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            message = f"expected an object for source {name!r} in {path}"
            raise TypeError(message)
        entry = cast("dict[str, object]", value)
        kind_text = _read_string(entry, "type", name)
        try:
            kind = SourceKind(kind_text)
        except ValueError as error:
            message = f"unsupported source type {kind_text!r} for {name!r}"
            raise ValueError(message) from error
        parsed: SnapshotEntry = {
            "type": kind,
            "rev": _read_string(entry, "rev", name),
            "hash": _read_string(entry, "hash", name),
        }
        if kind is SourceKind.github:
            parsed["owner"] = _read_string(entry, "owner", name)
            parsed["repo"] = _read_string(entry, "repo", name)
            if "fetch_submodules" in entry:
                fetch_submodules = entry["fetch_submodules"]
                if not isinstance(fetch_submodules, bool):
                    message = f"expected boolean 'fetch_submodules' in source {name!r}"
                    raise TypeError(message)
                parsed["fetch_submodules"] = fetch_submodules
        else:
            parsed["url"] = _read_string(entry, "url", name)
            source_root = entry.get("source_root")
            if source_root is not None:
                if not isinstance(source_root, str):
                    message = f"expected string 'source_root' in source {name!r}"
                    raise TypeError(message)
                parsed["source_root"] = source_root

        version = entry.get("version")
        if version is not None:
            if not isinstance(version, str):
                message = f"expected string 'version' in source {name!r}"
                raise TypeError(message)
            parsed["version"] = version
        snapshot[name] = parsed
    return snapshot


def read_manifest(path: Path) -> BuildManifest:
    """Read the fields used to plan a named build configuration."""
    raw = _read_object(path)
    manifest: BuildManifest = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            message = f"expected an object for configuration {name!r} in {path}"
            raise TypeError(message)
        configuration = cast("dict[str, object]", value)
        compatibility_value = configuration.get("compatibility")
        if not isinstance(compatibility_value, list):
            message = f"expected compatibility list in configuration {name!r}"
            raise TypeError(message)

        compatibility: list[CompatibilityEntry] = []
        compatibility_entries = cast("list[object]", compatibility_value)
        for index, entry_value in enumerate(compatibility_entries):
            if not isinstance(entry_value, dict):
                message = f"expected compatibility object at {name}[{index}]"
                raise TypeError(message)
            entry = cast("dict[str, object]", entry_value)
            context = f"configuration {name!r} compatibility entry {index}"
            compatibility.append(
                {
                    "consumer": _read_string(entry, "consumer", context),
                    "group": _read_string(entry, "group", context),
                    "isolation": _read_enum(Isolation, entry, "isolation", context),
                    "requirement": _read_enum(
                        RequirementAdapter,
                        entry,
                        "requirement",
                        context,
                    ),
                },
            )

        manifest[name] = {
            "packageName": _read_string(configuration, "packageName", name),
            "resolved": _read_string_list(configuration, "resolved", name),
            "compatibility": compatibility,
        }
    return manifest
