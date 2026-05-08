"""Primary-mesh and sidecar export helpers."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import NormalizedParams, RuntimeContext, stable_artifact_stem
from .errors import SetupFailure, ValidationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartArtifact:
    part_id: str
    mesh_path: Path
    bbox: dict[str, Any]
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecompositionArtifacts:
    primary_mesh_path: Path
    parts: tuple[PartArtifact, ...]
    segmentation: dict[str, Any]
    bboxes: dict[str, Any]
    completion: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_observability_record(
    *,
    context: RuntimeContext,
    params: NormalizedParams,
    artifact_paths: dict[str, Any],
    part_count: int,
) -> dict[str, Any]:
    return {
        "host": context.host_facts.to_dict(),
        "pipeline_stage": params.pipeline_stage,
        "output_mode": params.output_mode,
        "seed": params.seed,
        "part_count": part_count,
        "export_format": params.export_format,
        "artifact_paths": artifact_paths,
    }


def export_bundle(
    *,
    artifacts: DecompositionArtifacts,
    output_dir: Path,
    params: NormalizedParams,
    context: RuntimeContext,
) -> dict[str, Any]:
    source_suffix = artifacts.primary_mesh_path.suffix.lower().lstrip(".")
    if source_suffix != params.export_format:
        raise SetupFailure(
            "Primary mesh export format conversion is not implemented in the local runtime.",
            code="unsupported_primary_conversion",
            details={"source": source_suffix, "requested": params.export_format},
        )
    stem = stable_artifact_stem(artifacts.primary_mesh_path, params)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    expose_debug_parts = params.output_mode == "debug"
    if expose_debug_parts:
        parts_dir.mkdir(parents=True, exist_ok=True)
    primary_output = output_dir / f"{stem}.{params.export_format}"
    shutil.copy2(artifacts.primary_mesh_path, primary_output)
    copied_parts: list[dict[str, Any]] = []
    if expose_debug_parts:
        for part in sorted(artifacts.parts, key=lambda item: item.part_id):
            part_suffix = part.mesh_path.suffix.lower()
            if part_suffix != artifacts.primary_mesh_path.suffix.lower():
                raise ValidationError(
                    "All part meshes must match the primary mesh format for MVP exports.",
                    code="mismatched_part_format",
                    details={"part_id": part.part_id},
                )
            normalized_name = f"{part.part_id}{part_suffix}"
            destination = parts_dir / normalized_name
            shutil.copy2(part.mesh_path, destination)
            copied_parts.append(
                {
                    "part_id": part.part_id,
                    "path": str(destination.relative_to(output_dir)),
                    "bbox": part.bbox,
                    "label": part.label,
                    "metadata": part.metadata,
                }
            )
    segmentation_path = output_dir / "segmentation.json"
    bboxes_path = output_dir / "bboxes.json"
    _write_json(segmentation_path, artifacts.segmentation)
    _write_json(bboxes_path, artifacts.bboxes)
    completion_path: str | None = None
    if artifacts.completion is not None:
        completion_file = output_dir / "completion.json"
        _write_json(completion_file, artifacts.completion)
        completion_path = str(completion_file.relative_to(output_dir))
    artifact_paths: dict[str, Any] = {
        "primary_mesh": str(primary_output.relative_to(output_dir)),
        "parts_dir": str(parts_dir.relative_to(output_dir)) if expose_debug_parts else None,
        "segmentation": str(segmentation_path.relative_to(output_dir)),
        "bboxes": str(bboxes_path.relative_to(output_dir)),
        "completion": completion_path,
    }
    observability = build_observability_record(
        context=context,
        params=params,
        artifact_paths=artifact_paths,
        part_count=len(copied_parts),
    )
    bundle_manifest = {
        "primary_output": artifact_paths["primary_mesh"],
        "sidecars": {
            "parts": copied_parts,
            "segmentation": artifact_paths["segmentation"],
            "bboxes": artifact_paths["bboxes"],
            "completion": artifact_paths["completion"],
        },
        "routing_contract": {
            "primary_output_type": "mesh",
            "single_mesh_primary": True,
            "output_mode": params.output_mode,
            "parts_visibility": "debug_sidecars" if expose_debug_parts else "hidden_from_primary_outputs",
            "parts_are_debug_only": True,
            "semantic": False,
            "downstream_safe": ["UniRig", "Kimodo"],
        },
        "metadata": artifacts.metadata,
        "observability": observability,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    _write_json(manifest_path, bundle_manifest)
    LOGGER.info("hunyuan3d_part.export_bundle %s", json.dumps(observability, sort_keys=True))
    return {
        "primary_mesh": str(primary_output),
        "parts": copied_parts if expose_debug_parts else [],
        "segmentation": str(segmentation_path),
        "bboxes": str(bboxes_path),
        "completion": str(output_dir / completion_path) if completion_path else None,
        "bundle_manifest": str(manifest_path),
        "observability": observability,
    }
