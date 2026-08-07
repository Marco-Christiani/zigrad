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


class SnapshotEntry(TypedDict):
    rev: str
    hash: NotRequired[str]
    version: NotRequired[str]


type DependencySnapshot = dict[str, SnapshotEntry]


class CompatibilityEntry(TypedDict):
    consumer: str
    group: str
    isolation: str
    requirement: str


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


def _read_string_list(mapping: dict[str, object], field: str, context: str) -> list[str]:
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
        parsed: SnapshotEntry = {"rev": _read_string(entry, "rev", name)}
        for field in ("hash", "version"):
            optional = entry.get(field)
            if optional is not None:
                if not isinstance(optional, str):
                    message = f"expected string {field!r} in source {name!r}"
                    raise TypeError(message)
                parsed[field] = optional
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
                    "isolation": _read_string(entry, "isolation", context),
                    "requirement": _read_string(entry, "requirement", context),
                },
            )

        manifest[name] = {
            "packageName": _read_string(configuration, "packageName", name),
            "resolved": _read_string_list(configuration, "resolved", name),
            "compatibility": compatibility,
        }
    return manifest
