"""Primary-mesh and sidecar export helpers."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .comparison_report import build_comparison_report, build_comparison_summary
from .config import NormalizedParams, RuntimeContext, new_run_id, stable_artifact_stem
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


def _semantic_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
    return {
        "schema": report.get("schema"),
        "mode": report.get("mode"),
        "stage": report.get("stage"),
        "semantic": report.get("semantic"),
        "publishable": report.get("publishable"),
        "effective_part_count": report.get("effective_part_count"),
        "confidence": {
            "aggregate": confidence.get("aggregate"),
            "level": confidence.get("level"),
        },
        "warnings": list(report.get("warnings", [])[:5]) if isinstance(report.get("warnings"), list) else [],
    }


def _write_semantic_report_sidecar(
    *,
    output_dir: Path,
    analysis_dir: Path,
    metadata: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    semantic_report = metadata.get("semantic_report")
    if semantic_report is None:
        return None, None
    destination = analysis_dir / "semantic_report.json"
    root_alias = output_dir / "semantic_report.json"
    if isinstance(semantic_report, dict):
        _write_json(destination, semantic_report)
        shutil.copy2(destination, root_alias)
        return str(destination.relative_to(output_dir)), _semantic_report_summary(semantic_report)
    source = Path(str(semantic_report))
    if source.is_file():
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        shutil.copy2(destination, root_alias)
        try:
            loaded = json.loads(destination.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        summary = _semantic_report_summary(loaded) if isinstance(loaded, dict) else {"source": str(source)}
        return str(destination.relative_to(output_dir)), summary
    return None, {"warning": "semantic_report_metadata_present_but_unusable", "source": str(semantic_report)}


def _prepare_comparison_report_sidecar(
    *,
    artifacts: DecompositionArtifacts,
    params: NormalizedParams,
    analysis_dir: Path,
    output_dir: Path,
    run_id: str,
    stable_stem: str,
    output_stem: str,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    if params.semantic_resolver != "analysis":
        return None, None, artifacts.metadata
    semantic_report = artifacts.metadata.get("semantic_report")
    if not isinstance(semantic_report, dict):
        return None, None, artifacts.metadata

    destination = analysis_dir / "comparison_report.json"
    canonical_path = str(destination.relative_to(output_dir))
    comparison_report = build_comparison_report(
        params=params,
        artifacts=artifacts,
        semantic_report=semantic_report,
        image_evidence=artifacts.metadata.get("image_evidence"),
        canonical_path=canonical_path,
        run_id=run_id,
        stable_artifact_stem=stable_stem,
        output_stem=output_stem,
    )
    comparison_summary = build_comparison_summary(comparison_report)
    updated_semantic_report = {**semantic_report, "comparison_summary": comparison_summary}
    updated_metadata = {**artifacts.metadata, "semantic_report": updated_semantic_report}
    _write_json(destination, comparison_report)
    return canonical_path, comparison_report, updated_metadata


def build_observability_record(
    *,
    context: RuntimeContext,
    params: NormalizedParams,
    artifact_paths: dict[str, Any],
    part_count: int,
    stable_stem: str,
    run_id: str,
    output_stem: str,
) -> dict[str, Any]:
    return {
        "host": context.host_facts.to_dict(),
        "pipeline_stage": params.pipeline_stage,
        "output_mode": params.output_mode,
        "seed": params.seed,
        "part_count": part_count,
        "export_format": params.export_format,
        "stable_artifact_stem": stable_stem,
        "run_id": run_id,
        "output_stem": output_stem,
        "artifact_paths": artifact_paths,
    }


def export_bundle(
    *,
    artifacts: DecompositionArtifacts,
    output_dir: Path,
    params: NormalizedParams,
    context: RuntimeContext,
    run_id: str | None = None,
) -> dict[str, Any]:
    source_suffix = artifacts.primary_mesh_path.suffix.lower().lstrip(".")
    if source_suffix != params.export_format:
        raise SetupFailure(
            "Primary mesh export format conversion is not implemented in the local runtime.",
            code="unsupported_primary_conversion",
            details={"source": source_suffix, "requested": params.export_format},
        )
    stable_stem = stable_artifact_stem(artifacts.primary_mesh_path, params)
    effective_run_id = run_id or new_run_id()
    output_stem = f"{stable_stem}-{effective_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_dir / "analysis" / output_stem
    analysis_dir.mkdir(parents=True, exist_ok=True)
    parts_root = output_dir / "parts"
    parts_dir = parts_root / output_stem
    expose_debug_parts = params.output_mode == "debug"
    if expose_debug_parts:
        parts_dir.mkdir(parents=True, exist_ok=True)
    primary_output = output_dir / f"{output_stem}.{params.export_format}"
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
    segmentation_path = analysis_dir / "segmentation.json"
    bboxes_path = analysis_dir / "bboxes.json"
    _write_json(segmentation_path, artifacts.segmentation)
    _write_json(bboxes_path, artifacts.bboxes)
    shutil.copy2(segmentation_path, output_dir / "segmentation.json")
    shutil.copy2(bboxes_path, output_dir / "bboxes.json")
    completion_path: str | None = None
    if artifacts.completion is not None:
        completion_file = analysis_dir / "completion.json"
        _write_json(completion_file, artifacts.completion)
        shutil.copy2(completion_file, output_dir / "completion.json")
        completion_path = str(completion_file.relative_to(output_dir))
    else:
        (output_dir / "completion.json").unlink(missing_ok=True)
    comparison_report_path, comparison_report, metadata_for_sidecars = _prepare_comparison_report_sidecar(
        artifacts=artifacts,
        params=params,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        run_id=effective_run_id,
        stable_stem=stable_stem,
        output_stem=output_stem,
    )
    semantic_report_path, semantic_report_summary = _write_semantic_report_sidecar(
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        metadata=metadata_for_sidecars,
    )
    if semantic_report_path is None:
        (output_dir / "semantic_report.json").unlink(missing_ok=True)
    manifest_path = analysis_dir / "bundle_manifest.json"
    artifact_paths: dict[str, Any] = {
        "primary_mesh": str(primary_output.relative_to(output_dir)),
        "analysis_dir": str(analysis_dir.relative_to(output_dir)),
        "bundle_manifest": str(manifest_path.relative_to(output_dir)),
        "parts_dir": str(parts_dir.relative_to(output_dir)) if expose_debug_parts else None,
        "segmentation": str(segmentation_path.relative_to(output_dir)),
        "bboxes": str(bboxes_path.relative_to(output_dir)),
        "completion": completion_path,
        "semantic_report": semantic_report_path,
        "comparison_report": comparison_report_path,
    }
    if semantic_report_summary is not None:
        artifact_paths["semantic_summary"] = semantic_report_summary
    export_metadata = {
        **metadata_for_sidecars,
        "stable_artifact_stem": stable_stem,
        "run_id": effective_run_id,
        "output_stem": output_stem,
        "analysis_dir": str(analysis_dir.relative_to(output_dir)),
        "compatibility_aliases": {
            "latest_only": True,
            "paths": {
                "segmentation": "segmentation.json",
                "bboxes": "bboxes.json",
                "completion": "completion.json" if artifacts.completion is not None else None,
                "semantic_report": "semantic_report.json" if semantic_report_path else None,
                "bundle_manifest": "bundle_manifest.json",
            },
            "canonical_dir": str(analysis_dir.relative_to(output_dir)),
        },
    }
    observability = build_observability_record(
        context=context,
        params=params,
        artifact_paths=artifact_paths,
        part_count=len(copied_parts),
        stable_stem=stable_stem,
        run_id=effective_run_id,
        output_stem=output_stem,
    )
    bundle_manifest = {
        "primary_output": artifact_paths["primary_mesh"],
        "sidecars": {
            "parts": copied_parts,
            "segmentation": artifact_paths["segmentation"],
            "bboxes": artifact_paths["bboxes"],
            "completion": artifact_paths["completion"],
            "semantic_report": artifact_paths["semantic_report"],
            "comparison_report": artifact_paths["comparison_report"],
            "bundle_manifest": artifact_paths["bundle_manifest"],
        },
        "routing_contract": {
            "primary_output_type": "mesh",
            "single_mesh_primary": True,
            "output_mode": params.output_mode,
            "parts_visibility": "debug_sidecars" if expose_debug_parts else "hidden_from_primary_outputs",
            "parts_are_debug_only": True,
            "semantic": False,
            "downstream_safe": ["UniRig", "Kimodo"],
            "stable_artifact_stem": stable_stem,
            "run_id": effective_run_id,
            "output_stem": output_stem,
            "analysis_dir": artifact_paths["analysis_dir"],
            "analysis_sidecars_are_run_scoped": True,
            "root_sidecars_are_latest_aliases": True,
            "comparison_report_available": comparison_report_path is not None,
        },
        "metadata": export_metadata,
        "observability": observability,
    }
    _write_json(manifest_path, bundle_manifest)
    shutil.copy2(manifest_path, output_dir / "bundle_manifest.json")
    LOGGER.info("hunyuan3d_part.export_bundle %s", json.dumps(observability, sort_keys=True))
    return {
        "primary_mesh": str(primary_output),
        "parts": copied_parts if expose_debug_parts else [],
        "segmentation": str(segmentation_path),
        "bboxes": str(bboxes_path),
        "completion": str(output_dir / completion_path) if completion_path else None,
        "semantic_report": str(output_dir / semantic_report_path) if semantic_report_path else None,
        "comparison_report": str(output_dir / comparison_report_path) if comparison_report_path else None,
        "bundle_manifest": str(manifest_path),
        "analysis_dir": str(analysis_dir),
        "observability": observability,
        "metadata": export_metadata,
    }
