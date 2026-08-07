from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dependency_planner import (
    Candidate,
    Requirement,
    evaluate_requirement,
    plan_configuration,
    select_candidate,
)


class DependencyPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.xla = self.root / "xla"
        self.llvm = self.root / "llvm"
        self.tvm = self.root / "tvm"
        self.iree_llvm = self.root / "iree-llvm"

        self.write(
            self.xla / "third_party/llvm/workspace.bzl",
            'LLVM_COMMIT = "llvm-exact"\n',
        )
        self.write_llvm_version(self.llvm, 22)
        self.write_llvm_version(self.iree_llvm, 23)
        self.write(
            self.tvm / "cmake/modules/LLVM.cmake",
            "if (${TVM_LLVM_VERSION} LESS 60)\nendif()\n",
        )
        self.snapshot = {
            "llvm": {"rev": "llvm-exact"},
            "iree_llvm": {"rev": "iree-exact"},
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_llvm_version(self, source: Path, major: int) -> None:
        self.write(
            source / "cmake/Modules/LLVMVersion.cmake",
            "\n".join(
                (
                    f"set(LLVM_VERSION_MAJOR {major})",
                    "set(LLVM_VERSION_MINOR 0)",
                    "set(LLVM_VERSION_PATCH 0)",
                ),
            ),
        )

    def test_shared_exact_requirement_selects_candidate(self) -> None:
        manifest = {
            "shared": {
                "packageName": "zigrad-shared",
                "compatibility": [
                    {
                        "group": "host-llvm",
                        "consumer": "stablehlo-mlir",
                        "requirement": "xla-llvm",
                        "isolation": "in-process",
                    },
                    {
                        "group": "host-llvm",
                        "consumer": "tvm",
                        "requirement": "tvm-llvm",
                        "isolation": "in-process",
                    },
                ],
            },
        }
        plan = plan_configuration(
            "shared",
            manifest,
            self.snapshot,
            self.xla,
            self.llvm,
            self.tvm,
            self.iree_llvm,
        )

        self.assertTrue(plan["compatible"])
        self.assertEqual(plan["groups"][0]["candidate"]["revision"], "llvm-exact")
        self.assertEqual(
            plan["groups"][0]["candidate"]["selection"],
            "exact consumer requirement",
        )

    def test_standalone_range_uses_snapshot_policy(self) -> None:
        manifest = {
            "tvm": {
                "packageName": "zigrad-tvm",
                "compatibility": [
                    {
                        "group": "host-llvm",
                        "consumer": "tvm",
                        "requirement": "tvm-llvm",
                        "isolation": "in-process",
                    },
                ],
            },
        }
        plan = plan_configuration(
            "tvm",
            manifest,
            self.snapshot,
            self.xla,
            self.llvm,
            self.tvm,
            self.iree_llvm,
        )

        self.assertTrue(plan["compatible"])
        self.assertEqual(
            plan["groups"][0]["candidate"]["selection"],
            "Zigrad dependency snapshot policy",
        )

    def test_conflicting_exact_requirements_fail(self) -> None:
        requirements = [
            Requirement("a", "exact-revision", "a", revision="one"),
            Requirement("b", "exact-revision", "b", revision="two"),
        ]

        with self.assertRaisesRegex(ValueError, "conflicting exact LLVM revisions"):
            select_candidate(
                "host-llvm",
                requirements,
                self.snapshot,
                self.llvm,
                self.iree_llvm,
            )

    def test_minimum_version_can_reject_candidate(self) -> None:
        requirement = Requirement(
            "consumer",
            "minimum-version",
            "fixture",
            minimum_version=(23, 0, 0),
        )
        candidate = Candidate("llvm", "llvm-exact", (22, 0, 0), "fixture")

        result = evaluate_requirement(requirement, candidate, "llvm-exact")

        self.assertFalse(result.satisfied)


if __name__ == "__main__":
    unittest.main()
