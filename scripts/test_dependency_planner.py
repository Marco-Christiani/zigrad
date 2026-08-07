from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from dependency_prefetch import CommandOutput, complete_proposal, prefetch_source
from dependency_planner import (
    Candidate,
    Requirement,
    evaluate_requirement,
    plan_configuration,
    select_candidate,
)
from dependency_proposal import CandidateSource, propose_configuration
from dependency_schema import IntegrationRoot, SourceKind


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

    def test_iree_candidate_revision_must_be_materialized(self) -> None:
        manifest = {
            "iree": {
                "packageName": "zigrad-iree",
                "compatibility": [
                    {
                        "group": "iree-build-llvm",
                        "consumer": "iree",
                        "requirement": "iree-llvm",
                        "isolation": "out-of-process",
                    },
                ],
            },
        }

        plan = plan_configuration(
            "iree",
            manifest,
            self.snapshot,
            self.xla,
            self.llvm,
            self.tvm,
            self.iree_llvm,
            "iree-next",
        )

        self.assertFalse(plan["compatible"])
        self.assertEqual(
            plan["groups"][0]["candidate"]["revision"],
            "iree-next",
        )

    def test_xla_candidate_derives_companion_changes(self) -> None:
        self.write(
            self.xla / "third_party/llvm/workspace.bzl",
            "\n".join(
                (
                    'LLVM_COMMIT = "llvm-next"',
                    f'LLVM_SHA256 = "{"1" * 64}"',
                ),
            ),
        )
        self.write(
            self.xla / "third_party/stablehlo/workspace.bzl",
            "\n".join(
                (
                    'STABLEHLO_COMMIT = "stablehlo-next"',
                    f'STABLEHLO_SHA256 = "{"2" * 64}"',
                ),
            ),
        )
        self.write(
            self.xla / "tensorflow.bazelrc",
            (
                'build:pjrt_cuda12 --repo_env=HERMETIC_CUDA_VERSION="12.9.1" '
                '--repo_env=HERMETIC_CUDNN_VERSION="9.8.0" '
                '--repo_env=HERMETIC_NVSHMEM_VERSION="3.2.5"\n'
            ),
        )
        snapshot = {
            **self.snapshot,
            "xla": {"rev": "xla-current", "hash": "xla-hash"},
            "stablehlo": {"rev": "stablehlo-current", "hash": "stablehlo-hash"},
        }
        manifest = {
            "xla": {
                "packageName": "zigrad-xla",
                "resolved": ["stablehlo-mlir"],
            },
        }
        candidate = CandidateSource(IntegrationRoot.xla, self.xla, "xla-next")

        proposal = propose_configuration(
            "xla",
            manifest,
            snapshot,
            {IntegrationRoot.xla: candidate},
        )

        changed = {
            (entry["source"], entry["field"]): entry for entry in proposal["changes"]
        }
        self.assertEqual(changed[("llvm", "rev")]["proposed"], "llvm-next")
        self.assertEqual(
            changed[("stablehlo", "rev")]["proposed"],
            "stablehlo-next",
        )
        self.assertIsNone(changed[("xla", "hash")]["proposed"])
        self.assertFalse(proposal["complete"])
        self.assertEqual(proposal["constraints"]["xla"]["cuda"]["cuda"], "12.9.1")
        self.assertEqual(
            proposal["constraints"]["xla"]["companions"]["llvm"]["revision"],
            "llvm-next",
        )

    def test_iree_candidate_derives_gitlink_changes(self) -> None:
        snapshot = {
            **self.snapshot,
            "iree": {"rev": "iree-current", "hash": "iree-hash"},
            "iree_benchmark": {"rev": "benchmark-current", "hash": "hash"},
            "iree_flatcc": {"rev": "flatcc-current", "hash": "hash"},
            "iree_stablehlo": {"rev": "stablehlo-current", "hash": "hash"},
        }
        manifest = {
            "iree": {
                "packageName": "zigrad-iree",
                "resolved": ["iree"],
            },
        }
        candidate = CandidateSource(
            IntegrationRoot.iree,
            self.root / "iree",
            "iree-next",
        )
        revisions = iter(
            ("benchmark-next", "flatcc-next", "iree-exact", "stablehlo-next"),
        )

        with patch("dependency_proposal.gitlink_revision", side_effect=revisions):
            proposal = propose_configuration(
                "iree",
                manifest,
                snapshot,
                {IntegrationRoot.iree: candidate},
            )

        changed_revisions = {
            entry["source"]: entry["proposed"]
            for entry in proposal["changes"]
            if entry["field"] == "rev"
        }
        self.assertEqual(changed_revisions["iree_benchmark"], "benchmark-next")
        self.assertEqual(changed_revisions["iree_flatcc"], "flatcc-next")
        self.assertEqual(changed_revisions["iree_stablehlo"], "stablehlo-next")

    def test_candidate_must_be_demanded_by_configuration(self) -> None:
        manifest = {
            "core": {
                "packageName": "zigrad",
                "resolved": [],
            },
        }
        candidate = CandidateSource(IntegrationRoot.tvm, self.tvm, "tvm-next")

        with self.assertRaisesRegex(ValueError, "does not demand candidate roots"):
            propose_configuration(
                "core",
                manifest,
                self.snapshot,
                {IntegrationRoot.tvm: candidate},
            )

    def test_mirage_candidate_updates_revision_bearing_fields(self) -> None:
        revision = "mirage-current"
        snapshot = {
            **self.snapshot,
            "mirage": {
                "type": SourceKind.url,
                "rev": revision,
                "hash": "mirage-hash",
                "url": f"https://example.invalid/mirage-{revision}.tar.gz",
                "source_root": f"mirage-{revision}",
            },
        }
        manifest = {
            "mirage": {
                "packageName": "zigrad-mirage",
                "resolved": ["mirage-cuda"],
            },
        }
        candidate = CandidateSource(
            IntegrationRoot.mirage,
            self.root / "mirage",
            "mirage-next",
        )

        proposal = propose_configuration(
            "mirage",
            manifest,
            snapshot,
            {IntegrationRoot.mirage: candidate},
        )
        changes = {entry["field"]: entry["proposed"] for entry in proposal["changes"]}

        self.assertEqual(
            changes["url"], "https://example.invalid/mirage-mirage-next.tar.gz"
        )
        self.assertEqual(changes["source_root"], "mirage-mirage-next")

    def test_prefetch_completes_snapshot_without_writing_it(self) -> None:
        revision = "mirage-current"
        snapshot = {
            **self.snapshot,
            "mirage": {
                "type": SourceKind.url,
                "rev": revision,
                "hash": "mirage-hash",
                "url": f"https://example.invalid/mirage-{revision}.tar.gz",
                "source_root": f"mirage-{revision}",
            },
        }
        manifest = {
            "mirage": {
                "packageName": "zigrad-mirage",
                "resolved": ["mirage-cuda"],
            },
        }
        candidate = CandidateSource(
            IntegrationRoot.mirage,
            self.root / "mirage",
            "mirage-next",
        )
        proposal = propose_configuration(
            "mirage",
            manifest,
            snapshot,
            {IntegrationRoot.mirage: candidate},
        )
        commands: list[list[str]] = []

        def run(arguments: Sequence[str]) -> CommandOutput:
            commands.append(list(arguments))
            return CommandOutput('{"hash":"sha256-prefetched"}', "")

        completed = complete_proposal(snapshot, proposal, run)

        self.assertTrue(completed["complete"])
        self.assertEqual(completed["snapshot"]["mirage"]["hash"], "sha256-prefetched")
        self.assertEqual(snapshot["mirage"]["hash"], "mirage-hash")
        self.assertEqual(
            commands[0][-1], "https://example.invalid/mirage-mirage-next.tar.gz"
        )

    def test_submodule_prefetch_uses_fetchgit_hash(self) -> None:
        entry = {
            "type": SourceKind.github,
            "owner": "apache",
            "repo": "tvm",
            "rev": "tvm-next",
            "hash": "tvm-current-hash",
            "fetch_submodules": True,
        }
        commands: list[list[str]] = []

        def run(arguments: Sequence[str]) -> CommandOutput:
            command = list(arguments)
            commands.append(command)
            if command[0] == "nix-prefetch-git":
                return CommandOutput("", "hash is nix-base32-hash\n")
            return CommandOutput("sha256-prefetched\n", "")

        fetched_hash = prefetch_source(entry, run)

        self.assertEqual(fetched_hash, "sha256-prefetched")
        self.assertIn("--fetch-submodules", commands[0])
        self.assertEqual(commands[1][:3], ["nix", "hash", "convert"])


if __name__ == "__main__":
    unittest.main()
