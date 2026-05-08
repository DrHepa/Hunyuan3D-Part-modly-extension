"""Input validation for the mesh-first extension contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .config import ALLOWED_MESH_FORMATS, normalize_params
from .errors import ValidationError


@dataclass(frozen=True)
class ValidatedMeshRequest:
    mesh_path: Path
    mesh_format: str


def _dedupe_paths(paths: Sequence[str | Path] | None) -> list[Path]:
    if not paths:
        return []
    unique: list[Path] = []
    seen: set[Path] = set()
    for entry in paths:
        candidate = Path(entry)
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _has_parent_traversal(mesh_path: Path) -> bool:
    return any(part == ".." for part in mesh_path.parts)


def validate_mesh_path(
    mesh: str | Path,
    search_roots: Sequence[str | Path] | None = None,
) -> ValidatedMeshRequest:
    mesh_path = Path(mesh)
    suffix = mesh_path.suffix.lower().lstrip(".")
    if suffix not in ALLOWED_MESH_FORMATS:
        raise ValidationError(
            "Unsupported mesh format.",
            code="unsupported_mesh_format",
            details={"allowed": list(ALLOWED_MESH_FORMATS), "observed": suffix or None},
        )

    normalized_roots = _dedupe_paths(search_roots)

    if mesh_path.is_absolute():
        candidates = [mesh_path]
    else:
        if _has_parent_traversal(mesh_path):
            raise ValidationError(
                "Mesh input must stay within trusted roots.",
                code="invalid_mesh_path",
                details={
                    "observed": str(mesh_path),
                    "search_roots": [str(root) for root in normalized_roots],
                },
            )
        candidates = [mesh_path, *[root / mesh_path for root in normalized_roots]]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return ValidatedMeshRequest(mesh_path=candidate.resolve(), mesh_format=suffix)

    raise ValidationError(
        "Mesh input must point to an existing file.",
        code="missing_mesh_input",
        details={
            "mesh": str(mesh_path),
            "observed": str(mesh_path),
            "candidates": [str(candidate) for candidate in candidates],
            "search_roots": [str(root) for root in normalized_roots],
        },
    )


def validate_inputs(inputs: Mapping[str, object]) -> ValidatedMeshRequest:
    if "mesh" not in inputs:
        raise ValidationError(
            "Exactly one mesh input is required.",
            code="missing_mesh_input",
        )
    if len(inputs) != 1:
        raise ValidationError(
            "Exactly one mesh input is required.",
            code="unexpected_inputs",
            details={"observed_keys": sorted(inputs.keys())},
        )
    return validate_mesh_path(inputs["mesh"])


def validate_request(
    inputs: Mapping[str, object],
    params: Mapping[str, object] | None = None,
) -> tuple[ValidatedMeshRequest, dict[str, object]]:
    mesh_request = validate_inputs(inputs)
    normalized = normalize_params(dict(params or {}))
    return mesh_request, normalized.to_dict()
