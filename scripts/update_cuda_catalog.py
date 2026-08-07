"""Update exact CUDA redistributable entries from upstream manifests."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from dependency_metadata import (
    COMPONENT_SPECS,
    ComponentKind,
    ComponentSpec,
    CudaComponent,
    XlaCudaMetadata,
    hex_sha256_to_sri,
)

logger = logging.getLogger(__name__)


def fetch_json(url: str, timeout_seconds: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "zigrad-cuda-catalog/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def cuda_generation(cuda_version: str) -> str:
    return f"cuda{cuda_version.split('.')[0]}"


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        os.unlink(temporary_name)
        raise


class CudaCatalogResolver:
    """Resolves the complete Linux x86_64 CUDA artifact set used by Zigrad."""

    def resolve(
        self,
        cuda_version: str,
        cudnn_version: str,
        nvshmem_version: str,
        nccl: CudaComponent,
    ) -> dict[str, object]:
        selected_versions = {
            "cuda": cuda_version,
            "cudnn": cudnn_version,
            "nvshmem": nvshmem_version,
        }
        manifests: dict[tuple[str, str], dict[str, Any]] = {}
        components: dict[str, CudaComponent] = {}

        for spec in COMPONENT_SPECS:
            manifest_version = selected_versions[spec.version_source]
            cache_key = (spec.prefix, manifest_version)
            if cache_key not in manifests:
                url = f"{spec.prefix}redistrib_{manifest_version}.json"
                logger.info("fetching %s", url)
                manifests[cache_key] = fetch_json(url)

            components[spec.manifest_key] = self._component(
                manifests[cache_key],
                spec,
                cuda_version,
            )

        components["nccl"] = nccl
        return {
            "cudnn_version": cudnn_version,
            "nccl_version": nccl.version,
            "nvshmem_version": nvshmem_version,
            "components": {
                name: serialize_component(component)
                for name, component in sorted(components.items())
            },
        }

    @staticmethod
    def _component(
        manifest: dict[str, Any],
        spec: ComponentSpec,
        cuda_version: str,
    ) -> CudaComponent:
        try:
            entry = manifest[spec.manifest_key]
            platform = entry["linux-x86_64"]
        except KeyError as error:
            raise ValueError(
                f"{spec.manifest_key} lacks Linux x86_64 metadata",
            ) from error

        generation = cuda_generation(cuda_version)
        if generation in platform:
            platform = platform[generation]

        try:
            relative_path = platform["relative_path"]
            sha256_hex = platform["sha256"].lower()
            version = entry["version"]
        except KeyError as error:
            raise ValueError(f"incomplete metadata for {spec.manifest_key}") from error

        return CudaComponent(
            name=spec.manifest_key,
            version=version,
            url=spec.prefix + relative_path,
            sha256_hex=sha256_hex,
            hash_sri=hex_sha256_to_sri(sha256_hex),
            runtime_dir=spec.runtime_dir,
            kind=ComponentKind.tarball,
        )


def serialize_component(component: CudaComponent) -> dict[str, str]:
    return {
        "hash_sri": component.hash_sri,
        "kind": component.kind.value,
        "runtime_dir": component.runtime_dir,
        "sha256_hex": component.sha256_hex,
        "url": component.url,
        "version": component.version,
    }


def read_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cuda": {}}

    catalog = json.loads(path.read_text(encoding="utf-8"))
    if set(catalog) != {"cuda"} or not isinstance(catalog["cuda"], dict):
        raise ValueError(f"{path} is not a CUDA redistributable catalog")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update selected CUDA artifacts from NVIDIA manifests and XLA's NCCL map.",
    )
    parser.add_argument("--xla-src", type=Path, required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--bazel-config", default="pjrt_cuda12")
    parser.add_argument("--cudnn-version", required=True)
    parser.add_argument("--nvshmem-version", required=True)
    parser.add_argument("--nccl-version", required=True)
    parser.add_argument("--catalog", type=Path, default=Path("nix/cuda-redist.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    xla = XlaCudaMetadata(arguments.xla_src)
    catalog = read_catalog(arguments.catalog)
    resolver = CudaCatalogResolver()
    nccl = xla.nccl_wheel(arguments.cuda_version)
    if nccl.version != arguments.nccl_version:
        raise ValueError(
            f"requested NCCL {arguments.nccl_version}, but XLA metadata provides {nccl.version}",
        )

    logger.info(
        "resolving CUDA %s with cuDNN %s, NVSHMEM %s, and NCCL %s",
        arguments.cuda_version,
        arguments.cudnn_version,
        arguments.nvshmem_version,
        arguments.nccl_version,
    )
    catalog["cuda"][arguments.cuda_version] = resolver.resolve(
        arguments.cuda_version,
        arguments.cudnn_version,
        arguments.nvshmem_version,
        nccl,
    )

    write_json_atomic(arguments.catalog, catalog)
    logger.info("updated %s", arguments.catalog)


if __name__ == "__main__":
    main()
