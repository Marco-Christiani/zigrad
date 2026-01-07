#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_bazelrc(path):
    cfg_re = re.compile(r"^(common|build):([A-Za-z0-9_-]+)\s+(.*)$")
    env_re = re.compile(r"--repo_env=([A-Z0-9_]+)=\"([^\"]*)\"")
    link_re = re.compile(r"--config=([A-Za-z0-9_-]+)")

    available = []
    repo_env_direct = defaultdict(dict)
    config_links = defaultdict(list)
    line_count = 0
    matched_count = 0
    prefixed_count = 0
    colon_prefixed_count = 0
    prefix_samples = []
    colon_samples = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        line_count += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("common") or line.startswith("build"):
            prefixed_count += 1
            if len(prefix_samples) < 5:
                prefix_samples.append(line)
        if line.startswith("common:") or line.startswith("build:"):
            colon_prefixed_count += 1
            if len(colon_samples) < 5:
                colon_samples.append(line)
        m = cfg_re.match(line)
        if not m:
            continue
        matched_count += 1
        _, name, rest = m.groups()
        available.append(name)
        for key, val in env_re.findall(rest):
            repo_env_direct[name][key] = val
        for link in link_re.findall(rest):
            config_links[name].append(link)

    debug = {
        "line_count": line_count,
        "matched_count": matched_count,
        "prefixed_count": prefixed_count,
        "prefix_samples": prefix_samples,
        "colon_prefixed_count": colon_prefixed_count,
        "colon_samples": colon_samples,
    }

    return available, repo_env_direct, config_links, debug


def resolve(name, repo_env_direct, config_links, seen=None):
    if seen is None:
        seen = set()
    if name in seen:
        return {}
    seen.add(name)
    merged = {}
    for dep in config_links.get(name, []):
        merged.update(resolve(dep, repo_env_direct, config_links, seen))
    merged.update(repo_env_direct.get(name, {}))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Parse tensorflow.bazelrc configs.")
    parser.add_argument("--output", default="bazel-config.json")
    args = parser.parse_args()

    preferred = [
        Path("tensorflow.bazelrc"),
        Path("xla/tensorflow.bazelrc"),
        Path("third_party/xla/tensorflow.bazelrc"),
    ]

    files = []
    warnings = []

    def record_file(path):
        available, repo_env_direct, config_links, debug = parse_bazelrc(path)
        files.append(
            {
                "path": str(path),
                "config_count": len(set(available)),
                "available_configs": sorted(set(available)),
                "repo_env_direct": dict(repo_env_direct),
                "config_links": dict(config_links),
                "debug": debug,
            }
        )
        return available, repo_env_direct, config_links

    for candidate in preferred:
        if candidate.exists():
            record_file(candidate)

    if not files:
        for candidate in Path(".").rglob("*.bazelrc"):
            record_file(candidate)

    if not files:
        warnings.append("no bazelrc files found; check repository layout")

    def pick_source():
        if not files:
            return None
        non_empty = [f for f in files if f["config_count"] > 0]
        if non_empty:
            return max(non_empty, key=lambda f: f["config_count"])
        return files[0]

    source_file = pick_source()
    if source_file is None:
        available = []
        repo_env_direct = {}
        config_links = {}
    else:
        available = source_file["available_configs"]
        repo_env_direct = source_file["repo_env_direct"]
        config_links = source_file["config_links"]

    if source_file and source_file["config_count"] == 0:
        warnings.append(
            "bazelrc files found but no configs parsed; check parser assumptions"
        )

    definition_sources = defaultdict(list)
    for f in files:
        for name in f["available_configs"]:
            definition_sources[name].append(f["path"])

    repo_env_resolved = {
        name: resolve(name, repo_env_direct, config_links) for name in available
    }

    payload = {
        "source": source_file["path"] if source_file else None,
        "preferred": [str(p) for p in preferred],
        "files": files,
        "available_configs": sorted(set(available)),
        "definition_sources": dict(definition_sources),
        "repo_env_direct": dict(repo_env_direct),
        "repo_env_resolved": repo_env_resolved,
        "warnings": warnings,
    }

    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
