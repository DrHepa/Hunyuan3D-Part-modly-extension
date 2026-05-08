"""Stage dispatch helpers for Hunyuan3D-Part pipelines."""

from __future__ import annotations

from pathlib import Path

from .errors import RuntimeFailure
from .export import DecompositionArtifacts
from .p3_sam import ExecutionPlan, run_upstream_p3_sam
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
        return run_upstream_p3_sam(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=output_dir / ".stage-p3-sam",
        )
    if stage == "x-part":
        aabb_path = Path(plan.params.aabb_path) if plan.params.aabb_path else None
        return run_upstream_x_part(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=output_dir / ".stage-x-part",
            aabb_path=aabb_path,
        )
    if stage == "full":
        p3_sam_artifacts = run_upstream_p3_sam(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=output_dir / ".stage-p3-sam",
        )
        return run_upstream_x_part(
            plan,
            project_root=project_root,
            managed_python=managed_python,
            model_root=model_root,
            output_dir=output_dir / ".stage-x-part",
            aabb_path=_require_p3_sam_aabb(p3_sam_artifacts),
        )
    raise RuntimeFailure(
        "Unsupported pipeline stage reached runtime dispatch.",
        code="unsupported_pipeline_stage_dispatch",
        details={"stage": stage},
    )
