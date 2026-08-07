"""Explain LLVM selection for a Zigrad build configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dependency_planner import plan_configuration


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_plan(plan: dict[str, Any]) -> None:
    print(f"configuration: {plan['configuration']} ({plan['package']})")
    if not plan["groups"]:
        print("LLVM groups: none")
        return

    for group in plan["groups"]:
        candidate = group["candidate"]
        isolation = ", ".join(group["isolation"])
        status = "compatible" if group["compatible"] else "incompatible"
        print(f"\n{group['name']} [{isolation}]: {status}")
        print(
            f"  selected {candidate['name']}@{candidate['revision']} "
            f"(LLVM {candidate['version']})",
        )
        print(f"  selection: {candidate['selection']}")
        for requirement in group["requirements"]:
            marker = "ok" if requirement["satisfied"] else "fail"
            print(f"  [{marker}] {requirement['consumer']}: {requirement['explanation']}")
            print(f"       authority: {requirement['authority']}")
        print("  verification: not evaluated by the planner")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain LLVM compatibility for a Zigrad configuration.",
    )
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--xla-src", type=Path, required=True)
    parser.add_argument("--llvm-src", type=Path, required=True)
    parser.add_argument("--tvm-src", type=Path, required=True)
    parser.add_argument("--iree-llvm-src", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    plan = plan_configuration(
        arguments.configuration,
        read_json(arguments.manifest),
        read_json(arguments.snapshot),
        arguments.xla_src,
        arguments.llvm_src,
        arguments.tvm_src,
        arguments.iree_llvm_src,
    )
    if arguments.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print_plan(plan)

    if not plan["compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
