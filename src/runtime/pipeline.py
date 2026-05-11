"""Stage dispatch helpers for Hunyuan3D-Part pipelines."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .errors import RuntimeFailure
from .export import DecompositionArtifacts
from .p3_sam import ExecutionPlan, run_upstream_p3_sam
from .semantic_report import build_semantic_report, build_xpart_semantic_fallback
from .x_part import run_upstream_x_part


def _require_p3_sam_aabb(artifacts: DecompositionArtifacts) -> Path:
    aabb_path = artifacts.metadata.get("effective_aabb_path") or artifacts.metadata.get("aabb_path")
    if not aabb_path:
        raise RuntimeFailure(
            "P3-SAM completed but did not expose the AABB artifact required by the full pipeline.",
            code="missing_p3_sam_aabb",
            details={"metadata": artifacts.metadata},
        )
    resolved = Path(str(aabb_path))
    if not resolved.is_file():
        raise RuntimeFailure(
            "P3-SAM reported an AABB artifact path that does not exist.",
            code="missing_p3_sam_aabb",
            details={"aabb_path": str(resolved)},
        )
    return resolved


def _attach_semantic_report(
    artifacts: DecompositionArtifacts,
    *,
    stage: str,
    plan: ExecutionPlan,
    aabb_path: Path | None = None,
) -> DecompositionArtifacts:
    if plan.params.semantic_resolver != "analysis":
        return artifacts
    if stage == "x-part":
        semantic_report = build_xpart_semantic_fallback(plan.params, aabb_path, image_evidence=plan.image_evidence)
    else:
        effective_aabb_path = artifacts.metadata.get("effective_aabb_path") or artifacts.metadata.get("aabb_path")
        semantic_report = build_semantic_report(
            stage=stage,
            params=plan.params,
            segmentation=artifacts.segmentation,
            bboxes=artifacts.bboxes,
            metadata=artifacts.metadata,
            effective_aabb_path=effective_aabb_path,
            image_evidence=plan.image_evidence,
        )
    metadata = {**artifacts.metadata, "semantic_report": semantic_report, "image_evidence": plan.image_evidence}
    return replace(artifacts, metadata=metadata)


def _carry_semantic_report(
    artifacts: DecompositionArtifacts,
    *,
    source: DecompositionArtifacts,
) -> DecompositionArtifacts:
    semantic_report = source.metadata.get("semantic_report")
    if semantic_report is None:
        return artifacts
    metadata = {**artifacts.metadata, "semantic_report": semantic_report, "image_evidence": source.metadata.get("image_evidence")}
    return replace(artifacts, metadata=metadata)


def run_pipeline_stage(
    plan: ExecutionPlan,
    *,
    project_root: Path,
    managed_python: Path,
    model_root: Path | None,
    output_dir: Path,
) -> DecompositionArtifacts:
    stage = plan.params.pipeline_stage
    if stage == "p3-sam":
        p3_sam_output_dir = output_dir / ".stage-p3-sam" / plan.run_id
        artifacts = run_upstream_p3_sam(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=p3_sam_output_dir,
        )
        return _attach_semantic_report(artifacts, stage="p3-sam", plan=plan)
    if stage == "x-part":
        aabb_path = Path(plan.params.aabb_path) if plan.params.aabb_path else None
        x_part_output_dir = output_dir / ".stage-x-part" / plan.run_id
        artifacts = run_upstream_x_part(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=x_part_output_dir,
            aabb_path=aabb_path,
        )
        return _attach_semantic_report(artifacts, stage="x-part", plan=plan, aabb_path=aabb_path)
    if stage == "full":
        p3_sam_output_dir = output_dir / ".stage-p3-sam" / plan.run_id
        x_part_output_dir = output_dir / ".stage-x-part" / plan.run_id
        p3_sam_artifacts = run_upstream_p3_sam(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=p3_sam_output_dir,
        )
        p3_sam_artifacts = _attach_semantic_report(p3_sam_artifacts, stage="full", plan=plan)
        p3_sam_aabb_path = _require_p3_sam_aabb(p3_sam_artifacts)
        x_part_artifacts = run_upstream_x_part(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=x_part_output_dir,
            aabb_path=p3_sam_aabb_path,
        )
        return _carry_semantic_report(x_part_artifacts, source=p3_sam_artifacts)
    raise RuntimeFailure(
        "Unsupported pipeline stage reached runtime dispatch.",
        code="unsupported_pipeline_stage_dispatch",
        details={"stage": stage},
    )
