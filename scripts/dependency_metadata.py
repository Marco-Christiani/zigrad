"""Read upstream dependency metadata used by snapshot tools."""

from __future__ import annotations

import base64
import re
import urllib.request
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import cast


class ComponentKind(StrEnum):
    tarball = auto()
    wheel = auto()


@dataclass(frozen=True)
class CudaComponent:
    name: str
    version: str
    url: str
    sha256_hex: str
    hash_sri: str
    runtime_dir: str
    kind: ComponentKind


@dataclass(frozen=True)
class CudaDefaults:
    cuda: str
    cudnn: str
    nvshmem: str


@dataclass(frozen=True)
class SourceRequirement:
    revision: str
    archive_hash_sri: str


@dataclass(frozen=True)
class ComponentSpec:
    manifest_key: str
    runtime_dir: str
    prefix: str
    version_source: str


CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"

COMPONENT_SPECS = (
    ComponentSpec("cuda_cccl", "cuda_cccl", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_cudart", "cudart", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_nvrtc", "nvrtc", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcublas", "cublas", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcufft", "cufft", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_cupti", "cupti", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcusparse", "cusparse", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libnvjitlink", "nvjitlink", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_nvcc", "cuda_nvcc", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cudnn", "cudnn", CUDNN_REDIST_PREFIX, "cudnn"),
    ComponentSpec("libnvshmem", "nvshmem", NVSHMEM_REDIST_PREFIX, "nvshmem"),
)


def quoted_assignment(path: Path, name: str) -> str:
    pattern = re.compile(rf'^\s*{re.escape(name)}\s*=\s*"([^"]+)"\s*$')
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            return match.group(1)
    raise ValueError(f"{path} lacks {name}")


def llvm_version(source: Path) -> tuple[int, int, int]:
    cmake_path = source / "cmake/Modules/LLVMVersion.cmake"
    text = cmake_path.read_text(encoding="utf-8")
    values: list[int] = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(
            rf"set\(LLVM_VERSION_{component}\s+([0-9]+)\)",
            text,
        )
        if not match:
            raise ValueError(f"{cmake_path} lacks LLVM_VERSION_{component}")
        values.append(int(match.group(1)))
    return values[0], values[1], values[2]


def hex_sha256_to_sri(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"expected a 64-character SHA-256 value, got {value!r}")

    encoded = base64.b64encode(bytes.fromhex(normalized)).decode("ascii")
    return f"sha256-{encoded}"


def fetch_text(url: str, timeout_seconds: float = 30.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "zigrad-dependency-metadata/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return cast("bytes", response.read()).decode("utf-8")


class XlaMetadata:
    """Read dependency requirements from an XLA source tree."""

    def __init__(self, source: Path) -> None:
        self.source = source

    def llvm_revision(self) -> str:
        return quoted_assignment(
            self.source / "third_party/llvm/workspace.bzl",
            "LLVM_COMMIT",
        )

    def llvm_requirement(self) -> SourceRequirement:
        return self._source_requirement(
            self.source / "third_party/llvm/workspace.bzl",
            "LLVM",
        )

    def stablehlo_requirement(self) -> SourceRequirement:
        return self._source_requirement(
            self.source / "third_party/stablehlo/workspace.bzl",
            "STABLEHLO",
        )

    @staticmethod
    def _source_requirement(path: Path, prefix: str) -> SourceRequirement:
        return SourceRequirement(
            revision=quoted_assignment(path, f"{prefix}_COMMIT"),
            archive_hash_sri=hex_sha256_to_sri(
                quoted_assignment(path, f"{prefix}_SHA256"),
            ),
        )

    def defaults(self, bazel_config: str) -> CudaDefaults:
        bazelrc_path = self.source / "tensorflow.bazelrc"
        prefix = f"build:{bazel_config} "
        values: dict[str, str] = {}

        for line in bazelrc_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            for match in re.finditer(r'HERMETIC_(\w+)_VERSION="([^"]+)"', stripped):
                values[match.group(1).lower()] = match.group(2)

        missing = {"cuda", "cudnn", "nvshmem"} - values.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{bazelrc_path} lacks {names} for {bazel_config}")

        return CudaDefaults(
            cuda=values["cuda"],
            cudnn=values["cudnn"],
            nvshmem=values["nvshmem"],
        )

    def nccl_wheel(self, cuda_version: str) -> CudaComponent:
        metadata, source = self._rules_ml_cuda_metadata()
        dictionary_name = self._nccl_dictionary(metadata, cuda_version, source)
        dictionary = re.search(
            rf"{re.escape(dictionary_name)}\s*=\s*\{{(.*?)\n\}}",
            metadata,
            re.DOTALL,
        )
        if not dictionary:
            raise ValueError(f"NCCL dictionary {dictionary_name} not found in {source}")

        platform = re.search(
            r'"x86_64-unknown-linux-gnu"\s*:\s*\{([^}]+)\}',
            dictionary.group(1),
            re.DOTALL,
        )
        if not platform:
            raise ValueError(f"{dictionary_name} lacks an x86_64 Linux entry in {source}")

        fields = {
            name: match.group(1)
            for name in ("version", "url", "sha256")
            if (
                match := re.search(
                    rf'"{name}"\s*:\s*"([^"]+)"',
                    platform.group(1),
                )
            )
        }
        missing = {"version", "url", "sha256"} - fields.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{dictionary_name} lacks {names} in {source}")

        sha256_hex = fields["sha256"].lower()
        return CudaComponent(
            name="nccl",
            version=fields["version"],
            url=fields["url"],
            sha256_hex=sha256_hex,
            hash_sri=hex_sha256_to_sri(sha256_hex),
            runtime_dir="nccl",
            kind=ComponentKind.wheel,
        )

    def _rules_ml_cuda_metadata(self) -> tuple[str, str]:
        workspace_path = self.source / "WORKSPACE"
        workspace = workspace_path.read_text(encoding="utf-8")
        revision_match = re.search(
            r"rules_ml_toolchain-([0-9a-f]{40})",
            workspace,
        )
        if not revision_match:
            raise ValueError(f"rules_ml_toolchain revision not found in {workspace_path}")

        revision = revision_match.group(1)
        url = (
            "https://raw.githubusercontent.com/google-ml-infra/"
            f"rules_ml_toolchain/{revision}/third_party/gpus/cuda/hermetic/"
            "cuda_redist_versions.bzl"
        )
        return fetch_text(url), url

    @staticmethod
    def _nccl_dictionary(metadata: str, cuda_version: str, source: str) -> str:
        mapping = re.search(
            r"CUDA_NCCL_WHEELS\s*=\s*(\{.*?)(?:\n\n|# Ensures)",
            metadata,
            re.DOTALL,
        )
        if not mapping:
            raise ValueError(f"CUDA_NCCL_WHEELS not found in {source}")

        explicit = re.search(
            rf'"{re.escape(cuda_version)}"\s*:\s*([A-Za-z0-9_]+)\s*,?',
            mapping.group(1),
        )
        if explicit:
            return explicit.group(1)

        versions = re.search(
            r"CUDA_REDIST_JSON_DICT\s*=\s*\{(.*?)\n\}",
            metadata,
            re.DOTALL,
        )
        if not versions or not re.search(
            rf'"{re.escape(cuda_version)}"\s*:',
            versions.group(1),
        ):
            raise ValueError(f"CUDA {cuda_version} not found in {source}")

        major = cuda_version.split(".")[0]
        generated_name = f"CUDA_{major}_NCCL_WHEEL_DICT"
        if generated_name in metadata and re.search(
            rf'v:\s*{generated_name}\s+for\s+v\s+in\s+CUDA_REDIST_JSON_DICT\.keys\(\)\s+if\s+v\.startswith\("{major}"\)',
            mapping.group(1),
        ):
            return generated_name

        raise ValueError(f"NCCL mapping for CUDA {cuda_version} not found in {source}")
