"""Pure diagnostic comparison report builder.

The comparison report is deliberately non-authoritative. It compares runtime
facts that are already present in exported artifacts and never persists image
payloads, image hashes, image paths, or copied image files.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import DEFAULT_MAX_PARTS, NormalizedParams

COMPARISON_REPORT_SCHEMA = "hunyuan3d.comparison_report.v1"
NON_AUTHORITATIVE = True
PUBLISHABLE = False

_ALLOWED_IMAGE_INPUT_KEYS = ("present", "byte_size", "width", "height", "format", "mode")
_SEMANTIC_LIMITATIONS = (
    "no_ground_truth_labels",
    "no_2d_masks",
    "bbox_only_region_inference",
    "axis_assumption_z",
    "candidate_groups_are_heuristics_not_labels",
)


def _param_value(params: NormalizedParams | Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if params is None:
        return default
    if isinstance(params, Mapping):
        return params.get(key, default)
    return getattr(params, key, default)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sequence_len(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parts_count_from_payload(payload: Mapping[str, Any], *, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        observed = _safe_int(payload.get(key))
        if observed is not None:
            return observed
    for key in ("parts", "effective_parts"):
        observed = _sequence_len(payload.get(key))
        if observed is not None:
            return observed
    return None


def _completion_status(completion: Mapping[str, Any] | None) -> str | None:
    if not completion:
        return None
    status = completion.get("status")
    return str(status) if status is not None else None


def _resource_limits(completion: Mapping[str, Any] | None) -> Mapping[str, Any]:
    completion = completion or {}
    limits = completion.get("resource_limits")
    return limits if isinstance(limits, Mapping) else {}


def _count_summary(
    *,
    params: NormalizedParams | Mapping[str, Any] | None,
    artifacts: Any,
    semantic_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    segmentation = _mapping(getattr(artifacts, "segmentation", {}))
    bboxes = _mapping(getattr(artifacts, "bboxes", {}))
    completion = getattr(artifacts, "completion", None)
    completion_map = completion if isinstance(completion, Mapping) else None
    limits = _resource_limits(completion_map)
    semantic_report = semantic_report or {}

    requested = _safe_int(_param_value(params, "max_parts", None))
    effective_max = _safe_int(limits.get("effective_max_parts")) or _safe_int(segmentation.get("max_parts")) or requested
    raw_aabb = _safe_int(limits.get("raw_aabb_count")) or _safe_int(segmentation.get("raw_part_count"))
    effective_aabb = _safe_int(limits.get("effective_aabb_count")) or _safe_int(segmentation.get("effective_part_count"))
    if effective_aabb is None:
        effective_aabb = _parts_count_from_payload(bboxes, keys=("effective_part_count",))
    if raw_aabb is None:
        raw_aabb = _parts_count_from_payload(bboxes, keys=("raw_part_count",))

    exported_parts = len(tuple(getattr(artifacts, "parts", ()) or ()))
    semantic_effective = _safe_int(semantic_report.get("effective_part_count"))

    return {
        "requested_max_parts": requested,
        "effective_max_parts": effective_max,
        "raw_aabb_count": raw_aabb,
        "effective_aabb_count": effective_aabb,
        "exported_parts": exported_parts,
        "semantic_effective_part_count": semantic_effective,
        "completion_status": _completion_status(completion_map),
        "requested_max_parts_is_cap_not_produced_count": True,
    }


def _check(status: str, code: str, message: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "code": code, "message": message, "evidence": dict(evidence or {})}


def _build_checks(counts: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    requested = counts.get("requested_max_parts")
    exported = counts.get("exported_parts")
    raw_aabb = counts.get("raw_aabb_count")
    effective_aabb = counts.get("effective_aabb_count")
    semantic_effective = counts.get("semantic_effective_part_count")
    completion_status = counts.get("completion_status")

    if isinstance(requested, int) and isinstance(exported, int):
        if requested > exported:
            checks.append(
                _check(
                    "warning",
                    "requested_gt_exported",
                    "requested max_parts is a cap, not the produced part count; fewer parts were exported.",
                    {"requested_max_parts": requested, "exported_parts": exported},
                )
            )
        else:
            checks.append(_check("pass", "requested_gt_exported", "exported parts reached the requested cap.", {"requested_max_parts": requested, "exported_parts": exported}))
    else:
        checks.append(_check("not_applicable", "requested_gt_exported", "requested/exported counts are unavailable."))

    if isinstance(raw_aabb, int) and isinstance(effective_aabb, int):
        if raw_aabb > effective_aabb:
            checks.append(_check("warning", "raw_aabb_trimmed", "raw AABB guidance was trimmed to the effective count.", {"raw_aabb_count": raw_aabb, "effective_aabb_count": effective_aabb}))
        else:
            checks.append(_check("pass", "raw_aabb_trimmed", "raw and effective AABB counts match.", {"raw_aabb_count": raw_aabb, "effective_aabb_count": effective_aabb}))
    else:
        checks.append(_check("not_applicable", "raw_aabb_trimmed", "raw/effective AABB counts are unavailable."))

    if isinstance(exported, int) and isinstance(semantic_effective, int):
        if exported != semantic_effective:
            checks.append(_check("warning", "exported_ne_semantic_effective", "exported part count differs from semantic effective part count.", {"exported_parts": exported, "semantic_effective_part_count": semantic_effective}))
        else:
            checks.append(_check("pass", "exported_ne_semantic_effective", "exported and semantic effective counts match.", {"exported_parts": exported, "semantic_effective_part_count": semantic_effective}))
    else:
        checks.append(_check("not_applicable", "exported_ne_semantic_effective", "exported/semantic counts are unavailable."))

    if completion_status is None:
        checks.append(_check("warning", "completion_missing", "completion sidecar was not available for comparison."))
    elif completion_status == "completed":
        checks.append(_check("pass", "completion_status", "completion status is completed.", {"completion_status": completion_status}))
    else:
        checks.append(_check("warning", "completion_status", "completion status is not completed.", {"completion_status": completion_status}))

    checks.append(_check("pass", "privacy_metadata_only", "image evidence is metadata-only; no bytes, hashes, paths, base64, or copied images are stored."))
    return checks


def _image_evidence_summary(image_evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    source = _mapping(image_evidence or {})
    image_input = _mapping(source.get("image_input")) or source
    summary = {key: image_input[key] for key in _ALLOWED_IMAGE_INPUT_KEYS if key in image_input}
    summary["present"] = bool(summary.get("present", False))
    summary["persisted"] = False
    summary["metadata_only"] = True
    summary["non_authoritative"] = True
    summary["matching"] = {
        "method": "byte_size_exact" if summary.get("byte_size") is not None else "metadata_presence",
        "local_diagnostic_only": True,
        "cryptographic_identity": False,
        "visual_truth": False,
    }
    return summary


def _semantic_alignment(semantic_report: Mapping[str, Any] | None) -> dict[str, Any]:
    semantic_report = semantic_report or {}
    candidate_groups = semantic_report.get("candidate_groups") if isinstance(semantic_report.get("candidate_groups"), Mapping) else {}
    candidate_summary = {str(key): len(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else 0 for key, value in candidate_groups.items()}
    warnings = [str(item) for item in semantic_report.get("warnings", []) or []] if isinstance(semantic_report.get("warnings"), Sequence) and not isinstance(semantic_report.get("warnings"), (str, bytes, bytearray)) else []
    return {
        "non_authoritative": True,
        "publishable": False,
        "candidate_groups_are_labels": False,
        "candidate_groups_summary": candidate_summary,
        "confidence": semantic_report.get("confidence") if isinstance(semantic_report.get("confidence"), Mapping) else {"level": "low", "publishable": False},
        "limitations": list(_SEMANTIC_LIMITATIONS),
        "warnings": warnings,
    }


def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"


def _privacy_block() -> dict[str, Any]:
    return {
        "image_persisted": False,
        "stores_image_bytes": False,
        "stores_image_hash": False,
        "stores_image_path": False,
        "stores_image_base64": False,
        "copies_image_file": False,
    }


def build_comparison_report(
    *,
    params: NormalizedParams | Mapping[str, Any] | None,
    artifacts: Any,
    semantic_report: Mapping[str, Any] | None,
    image_evidence: Mapping[str, Any] | None = None,
    canonical_path: str | None = None,
    run_id: str | None = None,
    stable_artifact_stem: str | None = None,
    output_stem: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic diagnostic report without IO or input mutation."""
    counts = _count_summary(params=params, artifacts=artifacts, semantic_report=semantic_report)
    checks = _build_checks(counts)
    return {
        "schema": COMPARISON_REPORT_SCHEMA,
        "mode": "analysis",
        "publishable": PUBLISHABLE,
        "non_authoritative": NON_AUTHORITATIVE,
        "path": canonical_path,
        "run": {
            "run_id": run_id,
            "stable_artifact_stem": stable_artifact_stem,
            "output_stem": output_stem,
            "pipeline_stage": _param_value(params, "pipeline_stage"),
            "seed": _param_value(params, "seed"),
        },
        "counts": counts,
        "image_evidence": _image_evidence_summary(image_evidence),
        "checks": checks,
        "overall_status": _overall_status(checks),
        "semantic_alignment": _semantic_alignment(semantic_report),
        "privacy": _privacy_block(),
    }


def build_comparison_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("checks") if isinstance(report.get("checks"), Sequence) else []
    warnings = [str(check.get("code")) for check in checks if isinstance(check, Mapping) and str(check.get("status")) in {"warning", "fail"}]
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    key_counts = {
        "requested_max_parts": counts.get("requested_max_parts"),
        "effective_max_parts": counts.get("effective_max_parts"),
        "exported_parts": counts.get("exported_parts"),
        "semantic_effective_part_count": counts.get("semantic_effective_part_count"),
    }
    return {
        "path": report.get("path"),
        "overall_status": report.get("overall_status"),
        "warning_count": len(warnings),
        "warnings": warnings[:5],
        "key_counts": key_counts,
        "non_authoritative": True,
    }
