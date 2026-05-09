"""Diagnostic semantic report builder for regional part decomposition.

The report is intentionally sidecar-only: it summarizes existing P3-SAM/X-Part
artifacts and never mutates segmentation, AABB, or raw face-id inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import DEFAULT_MAX_PARTS, NormalizedParams

SEMANTIC_REPORT_SCHEMA = "hunyuan3d.semantic_report.v1"
SEMANTIC_REPORT_SOURCE = "semantic_decomposition_resolver"
SEMANTIC_REPORT_MODE = "analysis"
SEMANTIC_LEVEL = "region_inference"
FALLBACK_POLICY = "do_not_modify_xpart_inputs"
SUPPORTED_CANDIDATE_ROLES = (
    "head_region",
    "upper_body_region",
    "lower_body_region",
    "limb_region",
    "accessory_candidate",
    "unknown",
)
VERTICAL_AXIS = "z"


def _param_value(params: NormalizedParams | Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if params is None:
        return default
    if isinstance(params, Mapping):
        return params.get(key, default)
    return getattr(params, key, default)


def _as_float_triplet(value: object) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _bbox_descriptor(part: Mapping[str, Any]) -> dict[str, Any] | None:
    min_xyz = _as_float_triplet(part.get("min"))
    max_xyz = _as_float_triplet(part.get("max"))
    if min_xyz is None or max_xyz is None:
        bbox = part.get("bbox")
        if isinstance(bbox, Mapping):
            min_xyz = _as_float_triplet(bbox.get("min"))
            max_xyz = _as_float_triplet(bbox.get("max"))
    if min_xyz is None or max_xyz is None:
        return None
    ordered_min = [min(a, b) for a, b in zip(min_xyz, max_xyz, strict=True)]
    ordered_max = [max(a, b) for a, b in zip(min_xyz, max_xyz, strict=True)]
    center = [(a + b) / 2.0 for a, b in zip(ordered_min, ordered_max, strict=True)]
    extent = [max(b - a, 0.0) for a, b in zip(ordered_min, ordered_max, strict=True)]
    volume = extent[0] * extent[1] * extent[2]
    return {
        "min": ordered_min,
        "max": ordered_max,
        "center": center,
        "extent": extent,
        "volume": volume,
    }


def _part_id_index(part_id: object, fallback: int) -> int:
    if isinstance(part_id, int):
        return part_id
    if isinstance(part_id, str):
        try:
            return int(part_id.rsplit("-", 1)[-1])
        except ValueError:
            return fallback
    return fallback


def _face_count_index(segmentation: Mapping[str, Any]) -> dict[int, int]:
    indexed: dict[int, int] = {}
    for fallback, part in enumerate(segmentation.get("effective_parts", []) or []):
        if not isinstance(part, Mapping):
            continue
        part_index = _part_id_index(part.get("part_id"), fallback)
        try:
            indexed[part_index] = int(part.get("face_count", 0) or 0)
        except (TypeError, ValueError):
            indexed[part_index] = 0
    return indexed


def _raw_id_index(segmentation: Mapping[str, Any]) -> dict[int, int | None]:
    indexed: dict[int, int | None] = {}
    for fallback, part in enumerate(segmentation.get("effective_parts", []) or []):
        if not isinstance(part, Mapping):
            continue
        part_index = _part_id_index(part.get("part_id"), fallback)
        raw = part.get("raw_part_id")
        try:
            indexed[part_index] = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            indexed[part_index] = None
    return indexed


def _global_bbox(descriptors: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not descriptors:
        return None
    mins = [[float(item) for item in descriptor["min"]] for descriptor in descriptors]
    maxes = [[float(item) for item in descriptor["max"]] for descriptor in descriptors]
    return _bbox_descriptor(
        {
            "min": [min(values[axis] for values in mins) for axis in range(3)],
            "max": [max(values[axis] for values in maxes) for axis in range(3)],
        }
    )


def _normalized_position(descriptor: Mapping[str, Any], global_descriptor: Mapping[str, Any] | None) -> dict[str, Any]:
    if global_descriptor is None:
        return {
            "center": None,
            "extent": None,
            "vertical": None,
            "lateral": None,
        }
    global_min = [float(item) for item in global_descriptor["min"]]
    global_extent = [float(item) for item in global_descriptor["extent"]]
    center = [float(item) for item in descriptor["center"]]
    extent = [float(item) for item in descriptor["extent"]]

    normalized_center = [
        (center[axis] - global_min[axis]) / global_extent[axis] if global_extent[axis] else 0.5
        for axis in range(3)
    ]
    normalized_extent = [extent[axis] / global_extent[axis] if global_extent[axis] else 0.0 for axis in range(3)]
    lateral_denominator = global_extent[0] / 2.0
    lateral = abs(center[0] - float(global_descriptor["center"][0])) / lateral_denominator if lateral_denominator else 0.0
    return {
        "center": normalized_center,
        "extent": normalized_extent,
        "vertical": normalized_center[2],
        "lateral": lateral,
    }


def _role_candidate(
    *,
    normalized: Mapping[str, Any],
    volume_ratio: float,
    face_ratio: float,
    descriptor: Mapping[str, Any],
) -> tuple[str, float, list[str], list[str]]:
    vertical = float(normalized.get("vertical") if normalized.get("vertical") is not None else 0.5)
    lateral = float(normalized.get("lateral") if normalized.get("lateral") is not None else 0.0)
    extent = [float(item) for item in descriptor["extent"]]
    max_extent = max(extent) if extent else 0.0
    min_extent = min(extent) if extent else 0.0
    slender = bool(max_extent and min_extent / max_extent <= 0.45)
    evidence = [
        f"vertical_center={vertical:.3f}",
        f"lateral_offset={lateral:.3f}",
        f"volume_ratio={volume_ratio:.3f}",
        f"face_ratio={face_ratio:.3f}",
    ]
    warnings: list[str] = []

    if vertical >= 0.72 and lateral <= 0.55 and volume_ratio <= 0.35:
        return "head_region", 0.58, evidence + ["top_compact_centered_bbox"], warnings
    if 0.42 <= vertical <= 0.78 and lateral <= 0.55 and (volume_ratio >= 0.18 or face_ratio >= 0.25):
        return "upper_body_region", 0.55, evidence + ["central_mid_upper_large_part"], warnings
    if vertical <= 0.38 and (volume_ratio >= 0.10 or face_ratio >= 0.10):
        return "lower_body_region", 0.50, evidence + ["lower_vertical_region"], warnings
    if lateral >= 0.55 and slender:
        return "limb_region", 0.48, evidence + ["lateral_slender_bbox"], warnings
    if volume_ratio <= 0.08 and (lateral >= 0.45 or vertical >= 0.72 or vertical <= 0.25):
        return "accessory_candidate", 0.42, evidence + ["small_peripheral_bbox"], warnings

    warnings.append("ambiguous_region_candidate")
    return "unknown", 0.25, evidence + ["no_conservative_rule_matched"], warnings


def _candidate_groups(parts: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups = {role: [] for role in SUPPORTED_CANDIDATE_ROLES}
    for part in parts:
        role = str(part.get("primary_role", "unknown"))
        if role not in groups:
            role = "unknown"
        groups[role].append(str(part["effective_part_id"]))
    return groups


def _report_base(*, stage: str, params: NormalizedParams | Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "schema": SEMANTIC_REPORT_SCHEMA,
        "mode": SEMANTIC_REPORT_MODE,
        "source": SEMANTIC_REPORT_SOURCE,
        "stage": stage,
        "semantic": False,
        "semantic_level": SEMANTIC_LEVEL,
        "publishable": False,
        "fallback_policy": FALLBACK_POLICY,
        "max_parts": int(_param_value(params, "max_parts", DEFAULT_MAX_PARTS) or DEFAULT_MAX_PARTS),
    }


def build_semantic_report(
    *,
    stage: str,
    params: NormalizedParams | Mapping[str, Any] | None = None,
    segmentation: Mapping[str, Any] | None = None,
    bboxes: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    effective_aabb_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic, diagnostic-only regional semantic report."""
    segmentation = segmentation or {}
    bboxes = bboxes or {}
    metadata = metadata or {}
    face_counts = _face_count_index(segmentation)
    raw_ids = _raw_id_index(segmentation)
    bbox_parts = [part for part in bboxes.get("parts", []) or [] if isinstance(part, Mapping)]
    raw_part_count = int(segmentation.get("raw_part_count") or len(bbox_parts))
    effective_part_count = int(segmentation.get("effective_part_count") or len(bbox_parts))
    selection_policy = str(segmentation.get("selection_policy") or "bbox_order")

    descriptors_by_part: list[tuple[int, Mapping[str, Any], dict[str, Any]]] = []
    warnings: list[str] = []
    for fallback, bbox_part in enumerate(bbox_parts):
        descriptor = _bbox_descriptor(bbox_part)
        part_index = _part_id_index(bbox_part.get("part_id"), fallback)
        if descriptor is None:
            warnings.append(f"missing_bbox_descriptor:part-{part_index}")
            continue
        descriptors_by_part.append((part_index, bbox_part, descriptor))

    descriptors = [item[2] for item in descriptors_by_part]
    global_descriptor = _global_bbox(descriptors)
    total_volume = float(sum(descriptor["volume"] for descriptor in descriptors)) or 1.0
    total_faces = float(sum(face_counts.values())) or 1.0
    report_parts: list[dict[str, Any]] = []
    for part_index, bbox_part, descriptor in sorted(descriptors_by_part, key=lambda item: item[0]):
        normalized = _normalized_position(descriptor, global_descriptor)
        face_count = int(face_counts.get(part_index, bbox_part.get("face_count", 0) or 0))
        volume_ratio = float(descriptor["volume"]) / total_volume if total_volume else 0.0
        face_ratio = face_count / total_faces if total_faces else 0.0
        if len(descriptors_by_part) < 2:
            role = "unknown"
            confidence = 0.20
            evidence = ["single_part_report_is_ambiguous"]
            part_warnings = ["insufficient_parts_for_regional_inference"]
        else:
            role, confidence, evidence, part_warnings = _role_candidate(
                normalized=normalized,
                volume_ratio=volume_ratio,
                face_ratio=face_ratio,
                descriptor=descriptor,
            )
        effective_id = str(bbox_part.get("part_id") or f"part-{part_index}")
        report_parts.append(
            {
                "effective_part_id": effective_id,
                "raw_part_id": raw_ids.get(part_index, bbox_part.get("raw_part_id")),
                "bbox": descriptor,
                "face_count": face_count,
                "normalized": normalized,
                "candidate_roles": [{"role": role, "confidence": confidence}],
                "primary_role": role,
                "confidence": confidence,
                "evidence": evidence,
                "warnings": part_warnings,
            }
        )

    aggregate_confidence = round(sum(float(part["confidence"]) for part in report_parts) / len(report_parts), 3) if report_parts else 0.0
    report = _report_base(stage=stage, params=params)
    report.update(
        {
            "raw_part_count": raw_part_count,
            "effective_part_count": effective_part_count,
            "selection_policy": selection_policy,
            "candidate_groups": _candidate_groups(report_parts),
            "parts": report_parts,
            "confidence": {
                "aggregate": aggregate_confidence,
                "level": "medium" if aggregate_confidence >= 0.5 else "low",
                "publishable": False,
            },
            "diagnostics": {
                "axis_assumption": {"vertical_axis": VERTICAL_AXIS},
                "input_completeness": {
                    "has_segmentation": bool(segmentation),
                    "has_bboxes": bool(bbox_parts),
                    "has_face_counts": bool(face_counts),
                },
                "raw_arrays_omitted": ["face_ids", "raw_masks"],
                "mutates_xpart_inputs": False,
            },
            "warnings": warnings,
            "inputs": {
                "segmentation": _provenance_ref(segmentation, metadata, "segmentation"),
                "bboxes": _provenance_ref(bboxes, metadata, "bboxes"),
                "effective_aabb_path": str(effective_aabb_path or metadata.get("effective_aabb_path") or "") or None,
                "metadata_refs": _metadata_refs(metadata),
            },
        }
    )
    return report


def _provenance_ref(payload: Mapping[str, Any], metadata: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {
        "source": payload.get("source") or metadata.get("adapter") or "unknown",
        "path": metadata.get(f"{key}_path"),
        "embedded_summary_only": True,
    }


def _metadata_refs(metadata: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("adapter", "output_dir", "aabb_path", "effective_aabb_path", "raw_outputs")
    refs: dict[str, Any] = {}
    for key in allowed:
        value = metadata.get(key)
        if value is not None:
            refs[key] = value
    return refs


def _load_aabb_parts(aabb_path: Path) -> list[dict[str, Any]]:
    import numpy as np

    aabb = np.load(aabb_path)
    if aabb.ndim == 4 and aabb.shape[0] == 1:
        aabb = aabb[0]
    if aabb.ndim != 3 or tuple(aabb.shape[1:]) != (2, 3):
        raise ValueError(f"Expected AABB shape [K, 2, 3], got {aabb.shape!r}.")
    return [
        {"part_id": f"part-{index}", "min": part[0].tolist(), "max": part[1].tolist()}
        for index, part in enumerate(aabb)
    ]


def build_xpart_semantic_fallback(
    params: NormalizedParams | Mapping[str, Any] | None,
    aabb_path: str | Path | None,
) -> dict[str, Any]:
    """Build an AABB-only x-part report or deterministic unavailable warning."""
    if aabb_path is None:
        report = build_semantic_report(
            stage="x-part",
            params=params,
            segmentation={"source": "x-part", "selection_policy": "aabb_only_unavailable"},
            bboxes={"source": "x-part", "parts": []},
            metadata={},
        )
        report["warnings"].append("semantic_report_unavailable")
        report["diagnostics"]["fallback_reason"] = "missing_aabb_path"
        return report

    resolved = Path(aabb_path)
    try:
        parts = _load_aabb_parts(resolved)
    except Exception as exc:  # pragma: no cover - exact numpy errors vary by version
        report = build_semantic_report(
            stage="x-part",
            params=params,
            segmentation={"source": "x-part", "selection_policy": "aabb_only_unavailable"},
            bboxes={"source": "x-part", "parts": []},
            metadata={"aabb_path": str(resolved)},
        )
        report["warnings"].extend(["semantic_report_unavailable", "aabb_only_fallback_load_failed"])
        report["diagnostics"]["fallback_reason"] = type(exc).__name__
        return report

    max_parts = int(_param_value(params, "max_parts", DEFAULT_MAX_PARTS) or DEFAULT_MAX_PARTS)
    effective_parts = parts[:max_parts]
    report = build_semantic_report(
        stage="x-part",
        params=params,
        segmentation={
            "source": "x-part",
            "raw_part_count": len(parts),
            "effective_part_count": len(effective_parts),
            "max_parts": max_parts,
            "selection_policy": "aabb_order_limited",
        },
        bboxes={"source": "x-part", "parts": effective_parts},
        metadata={"aabb_path": str(resolved)},
        effective_aabb_path=resolved,
    )
    report["warnings"].append("aabb_only_semantic_report_limited")
    report["diagnostics"]["fallback_reason"] = "xpart_aabb_only"
    return report
