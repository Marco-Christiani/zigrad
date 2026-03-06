"""Unified version resolver for Zigrad external dependencies.

Resolves version pins from local source trees (XLA, IREE) and fetches CUDA
redistributable manifests from NVIDIA's CDN. Outputs a single nix/versions.json
consumed by Nix derivations.

Usage:
  # Resolve everything from local source trees
  python scripts/resolve-versions.py \
    --xla-src ./reference/xla \
    --iree-src ./reference/iree \
    --cuda-versions 12.8.1 12.9.1

  # Resolve just CUDA redist
  python scripts/resolve-versions.py --cuda-versions 12.8.1
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import json
import logging
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ComponentKind(StrEnum):
    tarball = auto()
    wheel = auto()


@dataclass(frozen=True)
class SourcePin:
    """A resolved source dependency pin."""

    commit: str
    urls: list[str]
    sha256_hex: str | None = None
    hash_sri: str | None = None


@dataclass(frozen=True)
class CudaComponent:
    """A single resolved CUDA redistributable component."""

    name: str
    version: str
    url: str
    sha256_hex: str
    hash_sri: str
    runtime_dir: str
    kind: ComponentKind


@dataclass(frozen=True)
class CudaDefaults:
    """Default CUDA/cuDNN/NVSHMEM versions parsed from XLA's bazelrc."""

    cuda: str
    cudnn: str
    nvshmem: str


@dataclass(frozen=True)
class XlaPin:
    """Resolved XLA source pins."""

    commit: str
    llvm: SourcePin
    stablehlo: SourcePin
    cuda_defaults: CudaDefaults


@dataclass(frozen=True)
class IreePin:
    """Resolved IREE submodule pins."""

    commit: str
    llvm_commit: str
    llvm_url: str
    stablehlo_commit: str
    stablehlo_url: str


@dataclass(frozen=True)
class CudaRedistPin:
    """Resolved CUDA redistributable bundle for a specific version."""

    cudnn_version: str
    nvshmem_version: str
    nccl_version: str | None
    components: dict[str, CudaComponent]


@dataclass(frozen=True)
class ComponentSpec:
    """Registry entry: which manifest to fetch, and how to map the result."""

    manifest_key: str
    runtime_dir: str
    prefix: str
    # Manifest version key (e.g. "cuda_version" means use the CUDA version,
    # "cudnn_version" means use the cuDNN version, etc.)
    version_source: str


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _hex_sha256_to_sri(hex_sha256: str) -> str:
    """Convert 64-hex sha256 into SRI string accepted by Nix fetchers."""
    h = hex_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise ValueError(f"Expected sha256 hex (64 chars), got: {hex_sha256!r}")
    raw = bytes.fromhex(h)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"sha256-{b64}"


def _http_get_json(url: str, *, timeout_s: float = 30.0) -> dict:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "zigrad-resolve/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _parse_bzl_assign(text: str, var: str) -> str:
    """Parse a simple VAR = "value" assignment from .bzl text."""
    pat = rf'^{re.escape(var)}\s*=\s*"([^"]+)"\s*$'
    for line in text.splitlines():
        m = re.match(pat, line.strip())
        if m:
            return m.group(1)
    raise RuntimeError(f"Failed to parse {var} from bzl text")


def _git_rev(repo: Path) -> str:
    """Get HEAD commit hash for a git repo, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _cuda_sub_key(cuda_version: str) -> str:
    """Derive the CUDA generation sub-key from a version string.

    NVIDIA redist manifests nest platform entries under a CUDA generation
    key (e.g. "cuda12", "cuda11") for components like cuDNN and NVSHMEM.
    """
    major = cuda_version.split(".")[0]
    return f"cuda{major}"


def _write_json(path: Path, obj: object) -> None:
    """Write JSON with deterministic key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CUDA redistributable component registry
#
# Unified spec: manifest key, runtime_dir name (matching build.zig rpaths),
# download URL prefix, and which version string selects the manifest.
# ---------------------------------------------------------------------------

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"

COMPONENT_SPECS: list[ComponentSpec] = [
    # Main CUDA redist components
    ComponentSpec("cuda_cudart", "cudart", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_nvrtc", "nvrtc", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcublas", "cublas", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcufft", "cufft", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_cupti", "cupti", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libcusparse", "cusparse", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("libnvjitlink", "nvjitlink", CUDA_REDIST_PREFIX, "cuda"),
    ComponentSpec("cuda_nvcc", "cuda_nvcc", CUDA_REDIST_PREFIX, "cuda"),
    # cuDNN (separate manifest, may have cuda-generation sub-key)
    ComponentSpec("cudnn", "cudnn", CUDNN_REDIST_PREFIX, "cudnn"),
    # NVSHMEM (separate manifest, may have cuda-generation sub-key)
    ComponentSpec("libnvshmem", "nvshmem", NVSHMEM_REDIST_PREFIX, "nvshmem"),
]


# ---------------------------------------------------------------------------
# XlaResolver
# ---------------------------------------------------------------------------


class XlaResolver:
    """Parse XLA source tree for LLVM, StableHLO, and CUDA version pins."""

    def __init__(self, xla_src: Path) -> None:
        self.xla_src = xla_src

    def resolve(self) -> XlaPin:
        return XlaPin(
            commit=_git_rev(self.xla_src),
            llvm=self._resolve_llvm(),
            stablehlo=self._resolve_stablehlo(),
            cuda_defaults=self._resolve_cuda_defaults(),
        )

    def resolve_nccl_wheel(self, *, cuda_version: str) -> CudaComponent | None:
        """Parse NCCL wheel info from cuda_redist_versions.bzl.

        The .bzl file contains a Starlark dict keyed by CUDA generation
        (CUDA_12_NCCL_WHEEL_DICT, CUDA_11_NCCL_WHEEL_DICT) with per-arch entries.
        We extract the x86_64 entry for the requested CUDA generation.
        """
        bzl_path = self.xla_src / "third_party/gpus/cuda/hermetic/cuda_redist_versions.bzl"
        if not bzl_path.exists():
            return None

        bzl = bzl_path.read_text()
        major = cuda_version.split(".")[0]

        # Find the CUDA_<N>_NCCL_WHEEL_DICT block for x86_64
        # The block we want is referenced by CUDA_NCCL_WHEELS[<ver>], which
        # points at CUDA_<major>_NCCL_WHEEL_DICT. Parse the dict directly.
        dict_name = f"CUDA_{major}_NCCL_WHEEL_DICT"
        dict_match = re.search(
            rf'{re.escape(dict_name)}\s*=\s*\{{(.*?)\n\}}',
            bzl,
            re.DOTALL,
        )
        if not dict_match:
            logger.warning("NCCL wheel dict %s not found in %s", dict_name, bzl_path)
            return None

        # Within that dict, find the x86_64 entry
        x86_block = re.search(
            r'"x86_64-unknown-linux-gnu"\s*:\s*\{([^}]+)\}',
            dict_match.group(1),
            re.DOTALL,
        )
        if not x86_block:
            return None

        block = x86_block.group(1)
        version_m = re.search(r'"version"\s*:\s*"([^"]+)"', block)
        url_m = re.search(r'"url"\s*:\s*"([^"]+)"', block)
        sha256_m = re.search(r'"sha256"\s*:\s*"([^"]+)"', block)

        if not (version_m and url_m and sha256_m):
            return None

        sha256_hex = sha256_m.group(1).lower()
        return CudaComponent(
            name="nccl",
            version=version_m.group(1),
            url=url_m.group(1),
            sha256_hex=sha256_hex,
            hash_sri=_hex_sha256_to_sri(sha256_hex),
            runtime_dir="nccl",
            kind=ComponentKind.wheel,
        )

    def _resolve_llvm(self) -> SourcePin:
        ws = (self.xla_src / "third_party/llvm/workspace.bzl").read_text()
        commit = _parse_bzl_assign(ws, "LLVM_COMMIT")
        sha256_hex = _parse_bzl_assign(ws, "LLVM_SHA256").lower()
        return SourcePin(
            commit=commit,
            urls=[
                f"https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz",
                f"https://github.com/llvm/llvm-project/archive/{commit}.tar.gz",
            ],
            sha256_hex=sha256_hex,
            hash_sri=_hex_sha256_to_sri(sha256_hex),
        )

    def _resolve_stablehlo(self) -> SourcePin:
        ws = (self.xla_src / "third_party/stablehlo/workspace.bzl").read_text()
        commit = _parse_bzl_assign(ws, "STABLEHLO_COMMIT")
        sha256_hex = _parse_bzl_assign(ws, "STABLEHLO_SHA256").lower()
        return SourcePin(
            commit=commit,
            urls=[
                f"https://github.com/openxla/stablehlo/archive/{commit}.zip",
            ],
            sha256_hex=sha256_hex,
            hash_sri=_hex_sha256_to_sri(sha256_hex),
        )

    def _resolve_cuda_defaults(self, bazelrc_config: str = "pjrt_cuda12") -> CudaDefaults:
        """Parse default CUDA/cuDNN/NVSHMEM versions from tensorflow.bazelrc.

        The config name (e.g. "pjrt_cuda12") must match what the Nix build
        uses in xla-pjrt-runtime-bazel.nix (--config=pjrt_cuda12).
        """
        bazelrc = (self.xla_src / "tensorflow.bazelrc").read_text()
        prefix = f"build:{bazelrc_config} "

        env_vars: dict[str, str] = {}
        for line in bazelrc.splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            for m in re.finditer(r'HERMETIC_(\w+)_VERSION="([^"]+)"', stripped):
                env_vars[m.group(1).lower()] = m.group(2)

        return CudaDefaults(
            cuda=env_vars.get("cuda", ""),
            cudnn=env_vars.get("cudnn", ""),
            nvshmem=env_vars.get("nvshmem", ""),
        )


# ---------------------------------------------------------------------------
# IreeResolver
# ---------------------------------------------------------------------------


class IreeResolver:
    """Resolve IREE submodule pins from a local IREE checkout."""

    def __init__(self, iree_src: Path) -> None:
        self.iree_src = iree_src

    def resolve(self) -> IreePin:
        iree_commit = _git_rev(self.iree_src)
        llvm_commit = self._submodule_commit("third_party/llvm-project")
        stablehlo_commit = self._submodule_commit("third_party/stablehlo")

        return IreePin(
            commit=iree_commit,
            llvm_commit=llvm_commit,
            llvm_url=f"github:iree-org/llvm-project/{llvm_commit}",
            stablehlo_commit=stablehlo_commit,
            stablehlo_url=f"github:iree-org/stablehlo/{stablehlo_commit}",
        )

    def _submodule_commit(self, path: str) -> str:
        """Get the pinned commit of a submodule from git ls-tree."""
        try:
            output = subprocess.check_output(
                ["git", "-C", str(self.iree_src), "ls-tree", "HEAD", path],
                text=True,
            ).strip()
            # Format: <mode> commit <hash>\t<path>
            parts = output.split()
            if len(parts) >= 3:
                return parts[2]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return "unknown"


# ---------------------------------------------------------------------------
# CudaRedistResolver
# ---------------------------------------------------------------------------


class CudaRedistResolver:
    """Fetch NVIDIA redist manifests and extract linux-x86_64 component URLs+hashes."""

    def resolve(
        self,
        cuda_version: str,
        *,
        cudnn_version: str,
        nvshmem_version: str,
        nccl: CudaComponent | None = None,
    ) -> CudaRedistPin:
        version_map = {
            "cuda": cuda_version,
            "cudnn": cudnn_version,
            "nvshmem": nvshmem_version,
        }

        # Group specs by (prefix, version_source) to batch manifest fetches.
        manifest_cache: dict[str, dict] = {}
        components: dict[str, CudaComponent] = {}

        for spec in COMPONENT_SPECS:
            manifest_version = version_map[spec.version_source]
            cache_key = f"{spec.prefix}|{manifest_version}"

            if cache_key not in manifest_cache:
                url = f"{spec.prefix}redistrib_{manifest_version}.json"
                logger.info("Fetching manifest: %s", url)
                manifest_cache[cache_key] = _http_get_json(url)

            manifest = manifest_cache[cache_key]
            comp = self._extract_component(
                manifest, spec, cuda_version=cuda_version,
            )
            if comp:
                components[spec.manifest_key] = comp

        if nccl:
            components["nccl"] = nccl

        return CudaRedistPin(
            cudnn_version=cudnn_version,
            nvshmem_version=nvshmem_version,
            nccl_version=nccl.version if nccl else None,
            components=components,
        )

    def _extract_component(
        self,
        manifest: dict,
        spec: ComponentSpec,
        *,
        cuda_version: str,
    ) -> CudaComponent | None:
        entry = manifest.get(spec.manifest_key)
        if not entry:
            logger.warning("Component %s not found in manifest", spec.manifest_key)
            return None

        linux = entry.get("linux-x86_64")
        if not linux:
            logger.warning("No linux-x86_64 for %s", spec.manifest_key)
            return None

        # Some manifests (cuDNN, NVSHMEM) nest under a CUDA generation sub-key
        sub_key = _cuda_sub_key(cuda_version)
        if sub_key in linux and isinstance(linux[sub_key], dict):
            linux = linux[sub_key]

        relative_path = linux["relative_path"]
        sha256_hex = linux["sha256"].lower()
        version = entry.get("version", "unknown")

        return CudaComponent(
            name=spec.manifest_key,
            version=version,
            url=spec.prefix + relative_path,
            sha256_hex=sha256_hex,
            hash_sri=_hex_sha256_to_sri(sha256_hex),
            runtime_dir=spec.runtime_dir,
            kind=ComponentKind.tarball,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_source_pin(pin: SourcePin) -> dict:
    return dataclasses.asdict(pin)


def _serialize_xla(xla: XlaPin) -> dict:
    return {
        "commit": xla.commit,
        "llvm": _serialize_source_pin(xla.llvm),
        "stablehlo": _serialize_source_pin(xla.stablehlo),
        "cuda_defaults": dataclasses.asdict(xla.cuda_defaults),
    }


def _serialize_iree(iree: IreePin) -> dict:
    return {
        "commit": iree.commit,
        "llvm": {"commit": iree.llvm_commit, "url": iree.llvm_url},
        "stablehlo": {"commit": iree.stablehlo_commit, "url": iree.stablehlo_url},
    }


def _serialize_component(comp: CudaComponent) -> dict:
    return {
        "version": comp.version,
        "url": comp.url,
        "sha256_hex": comp.sha256_hex,
        "hash_sri": comp.hash_sri,
        "runtime_dir": comp.runtime_dir,
        "kind": comp.kind.value,
    }


def _serialize_cuda_redist(pin: CudaRedistPin) -> dict:
    return {
        "cudnn_version": pin.cudnn_version,
        "nvshmem_version": pin.nvshmem_version,
        "nccl_version": pin.nccl_version,
        "components": {k: _serialize_component(v) for k, v in pin.components.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified version resolver for Zigrad external dependencies.",
    )
    ap.add_argument(
        "--xla-src", type=Path, default=None,
        help="Path to local XLA source tree (e.g. ./reference/xla)",
    )
    ap.add_argument(
        "--iree-src", type=Path, default=None,
        help="Path to local IREE source tree (e.g. ./reference/iree)",
    )
    ap.add_argument(
        "--cuda-versions", nargs="+", default=[],
        help="CUDA versions to resolve (e.g. 12.8.1 12.9.1)",
    )
    ap.add_argument(
        "--cudnn-version", type=str, default=None,
        help="cuDNN version override (default: from XLA cuda_defaults or 9.8.0)",
    )
    ap.add_argument(
        "--nvshmem-version", type=str, default=None,
        help="NVSHMEM version override (default: from XLA cuda_defaults or 3.2.5)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("nix/versions.json"),
        help="Output path (default: nix/versions.json)",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    result: dict = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }

    # Resolve XLA pins
    xla_pin: XlaPin | None = None
    nccl: CudaComponent | None = None
    cudnn_version = args.cudnn_version or "9.8.0"
    nvshmem_version = args.nvshmem_version or "3.2.5"

    if args.xla_src:
        xla_resolver = XlaResolver(args.xla_src)
        xla_pin = xla_resolver.resolve()
        result["xla"] = _serialize_xla(xla_pin)

        # Use XLA's cuda_defaults for cuDNN/NVSHMEM versions if not overridden
        if not args.cudnn_version and xla_pin.cuda_defaults.cudnn:
            cudnn_version = xla_pin.cuda_defaults.cudnn
        if not args.nvshmem_version and xla_pin.cuda_defaults.nvshmem:
            nvshmem_version = xla_pin.cuda_defaults.nvshmem

    # Resolve IREE pins
    if args.iree_src:
        iree_pin = IreeResolver(args.iree_src).resolve()
        result["iree"] = _serialize_iree(iree_pin)

    # Resolve CUDA redist
    if args.cuda_versions:
        cuda_resolver = CudaRedistResolver()
        cuda_data: dict = {}
        for ver in args.cuda_versions:
            logger.info("Resolving CUDA %s (cuDNN=%s, NVSHMEM=%s)", ver, cudnn_version, nvshmem_version)
            # Resolve NCCL wheel per CUDA version if XLA source is available
            if xla_pin and args.xla_src:
                nccl = XlaResolver(args.xla_src).resolve_nccl_wheel(cuda_version=ver)
            redist_pin = cuda_resolver.resolve(
                ver,
                cudnn_version=cudnn_version,
                nvshmem_version=nvshmem_version,
                nccl=nccl,
            )
            cuda_data[ver] = _serialize_cuda_redist(redist_pin)
        result["cuda"] = cuda_data

    _write_json(args.out, result)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
