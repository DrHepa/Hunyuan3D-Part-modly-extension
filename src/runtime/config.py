"""Runtime configuration, host gates, and deterministic parameter normalization."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import CompatibilityFailure, DependencyFailure, HunyuanPartError, ValidationError

ALLOWED_MESH_FORMATS = ("glb", "obj", "stl", "ply")
SUPPORTED_EXPORT_FORMATS = ("glb",)
SUPPORTED_PIPELINE_STAGES = ("p3-sam", "x-part", "full")
SUPPORTED_OUTPUT_MODES = ("primary", "analysis", "debug")
SUPPORTED_QUALITY_PRESETS = ("fast", "balanced", "quality")
SUPPORTED_SEMANTIC_RESOLVERS = ("off", "analysis")
DEFAULT_PARAMS = {
    "pipeline_stage": "p3-sam",
    "export_format": "glb",
    "output_mode": "primary",
    "semantic_resolver": "off",
}
REQUIRED_RUNTIME_MODULES = ("numpy", "torch", "trimesh")
MINIMUM_PYTHON = (3, 10)
DEFAULT_MAX_PARTS = 32
MAX_PARTS_HARD_CAP = 512


XPART_RESOURCE_PROFILES: dict[str, dict[str, object]] = {
    "fast": {
        "num_inference_steps": 10,
        "octree_resolution": 256,
        "num_chunks": 8000,
        "surface_point_count": 8192,
        "bbox_point_count": 8192,
        "total_bbox_point_budget": 32768,
        "min_bbox_point_count": 4096,
        "torch_dtype": "float32",
    },
    "balanced": {
        "num_inference_steps": 30,
        "octree_resolution": 380,
        "num_chunks": 12000,
        "surface_point_count": 8192,
        "bbox_point_count": 8192,
        "total_bbox_point_budget": 65536,
        "min_bbox_point_count": 4096,
        "torch_dtype": "float32",
    },
    "quality": {
        "num_inference_steps": 50,
        "octree_resolution": 512,
        "num_chunks": 20000,
        "surface_point_count": 8192,
        "bbox_point_count": 8192,
        "total_bbox_point_budget": 98304,
        "min_bbox_point_count": 4096,
        "torch_dtype": "float32",
    },
}


@dataclass(frozen=True)
class HostFacts:
    os_name: str
    arch: str
    python_version: str
    python_abi: str
    cuda_visible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "os_name": self.os_name,
            "arch": self.arch,
            "python_version": self.python_version,
            "python_abi": self.python_abi,
            "cuda_visible": self.cuda_visible,
        }


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    model_root: Path
    artifacts_root: Path
    logs_root: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "model_root": str(self.model_root),
            "artifacts_root": str(self.artifacts_root),
            "logs_root": str(self.logs_root),
        }


@dataclass(frozen=True)
class NormalizedParams:
    pipeline_stage: str
    export_format: str
    output_mode: str = "primary"
    semantic_resolver: str = "off"
    max_parts: int | None = None
    seed: int | None = None
    quality_preset: str | None = None
    aabb_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "pipeline_stage": self.pipeline_stage,
            "export_format": self.export_format,
            "output_mode": self.output_mode,
            "semantic_resolver": self.semantic_resolver,
        }
        if self.max_parts is not None:
            payload["max_parts"] = self.max_parts
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.quality_preset is not None:
            payload["quality_preset"] = self.quality_preset
        if self.aabb_path is not None:
            payload["aabb_path"] = self.aabb_path
        return payload


@dataclass(frozen=True)
class SupportAssessment:
    ready: bool
    status: str
    warnings: tuple[str, ...] = ()
    failure: dict[str, object] | None = None
    dependencies: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ready": self.ready,
            "status": self.status,
            "warnings": list(self.warnings),
            "dependencies": self.dependencies,
        }
        if self.failure:
            payload["failure"] = self.failure
        return payload


@dataclass(frozen=True)
class RuntimeContext:
    host_facts: HostFacts
    paths: RuntimePaths
    support: SupportAssessment
    default_params: dict[str, str]


def resolve_host_facts(
    env: dict[str, str] | None = None,
    *,
    torch_cuda_probe: Callable[[], bool | None] | None = None,
) -> HostFacts:
    current_env = env or os.environ
    cuda_visible = bool(current_env.get("CUDA_HOME") or current_env.get("CUDA_PATH"))
    visible_devices = current_env.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip() not in ("", "-1"):
        cuda_visible = True
    probe = torch_cuda_probe or probe_torch_cuda_availability
    if not cuda_visible and probe():
        cuda_visible = True
    return HostFacts(
        os_name=platform.system().lower(),
        arch=platform.machine().lower(),
        python_version=platform.python_version(),
        python_abi=sys.abiflags or f"cp{sys.version_info.major}{sys.version_info.minor}",
        cuda_visible=cuda_visible,
    )


def probe_torch_cuda_availability() -> bool | None:
    try:
        module = importlib.import_module("torch")
    except Exception:
        return None
    cuda = getattr(module, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_available"):
        return None
    try:
        return bool(cuda.is_available())
    except Exception:
        return None


def resolve_runtime_paths(project_root: Path | None = None) -> RuntimePaths:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    model_root = infer_model_root(root)
    return RuntimePaths(
        project_root=root,
        model_root=model_root,
        artifacts_root=root / ".runtime" / "artifacts",
        logs_root=root / ".runtime" / "logs",
    )


def infer_model_root(project_root: Path) -> Path:
    root = project_root.resolve()
    if root.parent.name == "extensions":
        return root.parent.parent / "models" / root.name
    return root / ".runtime" / "models"


def inspect_dependencies(
    dependency_checker: Callable[[str], object | None] | None = None,
) -> dict[str, bool]:
    checker = dependency_checker or importlib.util.find_spec
    return {module: checker(module) is not None for module in REQUIRED_RUNTIME_MODULES}


def evaluate_host_support(
    host_facts: HostFacts,
    *,
    dependency_checker: Callable[[str], object | None] | None = None,
) -> SupportAssessment:
    warnings: list[str] = []
    dependencies = inspect_dependencies(dependency_checker)
    version_parts = host_facts.python_version.split(".")
    version_tuple = tuple(int(part) for part in version_parts[:2])
    if host_facts.os_name == "linux" and host_facts.arch in {"aarch64", "arm64"}:
        warnings.append("linux_arm64_risk=elevated")
    if version_tuple < MINIMUM_PYTHON:
        return SupportAssessment(
            ready=False,
            status="blocked",
            failure=CompatibilityFailure(
                "Python ABI is below the minimum supported runtime floor.",
                code="unsupported_python_abi",
                details={"required": ">=3.10", "observed": host_facts.python_version},
            ).to_dict(),
            dependencies=dependencies,
        )
    if not host_facts.cuda_visible:
        warnings.append("cpu_fallback_not_advertised")
        return SupportAssessment(
            ready=False,
            status="blocked",
            warnings=tuple(warnings),
            failure=CompatibilityFailure(
                "CUDA visibility is required; CPU fallback is intentionally not advertised.",
                code="cuda_not_visible",
                details=host_facts.to_dict(),
            ).to_dict(),
            dependencies=dependencies,
        )
    missing = [name for name, present in dependencies.items() if not present]
    if missing:
        return SupportAssessment(
            ready=False,
            status="blocked",
            warnings=tuple(warnings),
            failure=DependencyFailure(
                "Required runtime dependencies are missing.",
                details={"missing": missing},
            ).to_dict(),
            dependencies=dependencies,
        )
    return SupportAssessment(
        ready=True,
        status="ready_with_warnings" if warnings else "ready",
        warnings=tuple(warnings),
        dependencies=dependencies,
    )


def resolve_runtime_context(
    *,
    project_root: Path | None = None,
    env: dict[str, str] | None = None,
    dependency_checker: Callable[[str], object | None] | None = None,
    host_facts: HostFacts | None = None,
) -> RuntimeContext:
    facts = host_facts or resolve_host_facts(env)
    paths = resolve_runtime_paths(project_root)
    support = evaluate_host_support(facts, dependency_checker=dependency_checker)
    return RuntimeContext(
        host_facts=facts,
        paths=paths,
        support=support,
        default_params=dict(DEFAULT_PARAMS),
    )


def normalize_params(raw_params: dict[str, object] | None = None) -> NormalizedParams:
    params = dict(DEFAULT_PARAMS)
    if raw_params:
        params.update(raw_params)
    pipeline_stage = str(params["pipeline_stage"]).strip().lower()
    if pipeline_stage not in SUPPORTED_PIPELINE_STAGES:
        raise ValidationError(
            "Unsupported pipeline stage.",
            code="unsupported_pipeline_stage",
            details={"allowed": list(SUPPORTED_PIPELINE_STAGES), "observed": pipeline_stage},
        )
    export_format = str(params["export_format"]).strip().lower()
    if export_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValidationError(
            "Unsupported export format.",
            code="unsupported_export_format",
            details={"allowed": list(SUPPORTED_EXPORT_FORMATS), "observed": export_format},
        )
    output_mode = str(params["output_mode"]).strip().lower()
    if output_mode not in SUPPORTED_OUTPUT_MODES:
        raise ValidationError(
            "output_mode must be one of the supported visibility contracts.",
            code="invalid_output_mode",
            details={"allowed": list(SUPPORTED_OUTPUT_MODES), "observed": output_mode},
        )
    semantic_resolver = str(params["semantic_resolver"]).strip().lower()
    if semantic_resolver not in SUPPORTED_SEMANTIC_RESOLVERS:
        raise ValidationError(
            "semantic_resolver must be one of the active non-mutating resolver modes; guided is reserved for a future gated workflow.",
            code="invalid_semantic_resolver",
            details={"allowed": list(SUPPORTED_SEMANTIC_RESOLVERS), "observed": semantic_resolver},
        )
    max_parts = params.get("max_parts")
    if max_parts is None:
        max_parts = DEFAULT_MAX_PARTS
    else:
        try:
            max_parts = int(max_parts)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "max_parts must be an integer.",
                code="invalid_max_parts",
                details={"observed": max_parts},
            ) from exc
        if max_parts < 1:
            raise ValidationError(
                "max_parts must be positive.",
                code="invalid_max_parts",
                details={"observed": max_parts},
            )
        if max_parts > MAX_PARTS_HARD_CAP:
            raise ValidationError(
                "max_parts exceeds the hard runtime safety cap.",
                code="resource_guardrail_exceeded",
                details={"observed": max_parts, "maximum": MAX_PARTS_HARD_CAP},
            )
    seed = params.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "seed must be an integer.",
                code="invalid_seed",
                details={"observed": seed},
            ) from exc
    quality_preset = params.get("quality_preset")
    if quality_preset is None:
        quality_preset = "balanced"
    else:
        quality_preset = str(quality_preset).strip().lower()
        if quality_preset not in SUPPORTED_QUALITY_PRESETS:
            raise ValidationError(
                "quality_preset must be one of the supported X-Part resource profiles.",
                code="invalid_quality_preset",
                details={"allowed": list(SUPPORTED_QUALITY_PRESETS), "observed": quality_preset},
            )
    aabb_path = params.get("aabb_path")
    if aabb_path is not None:
        aabb_path = str(aabb_path).strip()
        if not aabb_path:
            raise ValidationError(
                "aabb_path cannot be empty when provided.",
                code="invalid_aabb_path",
            )
    return NormalizedParams(
        pipeline_stage=pipeline_stage,
        export_format=export_format,
        output_mode=output_mode,
        semantic_resolver=semantic_resolver,
        max_parts=max_parts,
        seed=seed,
        quality_preset=quality_preset,
        aabb_path=aabb_path,
    )


def resolve_x_part_resource_limits(params: NormalizedParams) -> dict[str, object]:
    """Return explicit X-Part runtime knobs matching Modly GPU execution patterns.

    The upstream X-Part defaults leave several heavyweight kwargs implicit
    (`num_chunks=400000`, 81920 sampled points per surface/part, float32
    module placement, and ignored `max_parts`).
    Modly extensions that behave well pass device/dtype and chunking explicitly,
    so this helper keeps those values visible and testable.
    """

    preset = params.quality_preset or "balanced"
    profile = XPART_RESOURCE_PROFILES[preset]
    effective_max_parts = params.max_parts if params.max_parts is not None else DEFAULT_MAX_PARTS
    if effective_max_parts > MAX_PARTS_HARD_CAP:
        raise ValidationError(
            "max_parts exceeds the hard runtime safety cap.",
            code="resource_guardrail_exceeded",
            details={"observed": effective_max_parts, "maximum": MAX_PARTS_HARD_CAP},
        )
    surface_point_count = int(profile["surface_point_count"])
    requested_bbox_point_count = int(profile["bbox_point_count"])
    total_bbox_point_budget = int(profile["total_bbox_point_budget"])
    min_bbox_point_count = int(profile["min_bbox_point_count"])
    adaptive_bbox_point_count = min(
        requested_bbox_point_count,
        max(min_bbox_point_count, total_bbox_point_budget // max(1, int(effective_max_parts))),
    )
    return {
        "quality_preset": preset,
        "effective_max_parts": int(effective_max_parts),
        "max_parts_hard_cap": MAX_PARTS_HARD_CAP,
        "num_inference_steps": int(profile["num_inference_steps"]),
        "octree_resolution": int(profile["octree_resolution"]),
        "num_chunks": int(profile["num_chunks"]),
        "surface_point_count": surface_point_count,
        "bbox_point_count": adaptive_bbox_point_count,
        "requested_bbox_point_count": requested_bbox_point_count,
        "total_bbox_point_budget": total_bbox_point_budget,
        "min_bbox_point_count": min_bbox_point_count,
        "point_budget_policy": "cap_total_bbox_points_by_effective_max_parts",
        "torch_dtype": str(profile["torch_dtype"]),
    }


def stable_artifact_stem(mesh_path: Path, params: NormalizedParams) -> str:
    seed_part = params.seed if params.seed is not None else "noseed"
    digest = hashlib.sha1(
        f"{mesh_path.stem}|{params.pipeline_stage}|{params.export_format}|{seed_part}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{mesh_path.stem}-{params.pipeline_stage}-{seed_part}-{digest}"


def raise_for_support(assessment: SupportAssessment) -> None:
    if assessment.ready:
        return
    failure = assessment.failure or {}
    category = failure.get("category")
    message = str(failure.get("message", "Runtime support blocked."))
    details = dict(failure.get("details", {}))
    code = str(failure.get("code", "support_blocked"))
    error_map: dict[str, type[HunyuanPartError]] = {
        "compatibility_failure": CompatibilityFailure,
        "dependency_failure": DependencyFailure,
    }
    error_type = error_map.get(str(category), CompatibilityFailure)
    raise error_type(message, code=code, details=details)
