"""P3-SAM runtime planning and upstream subprocess adapter."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .config import DEFAULT_MAX_PARTS, NormalizedParams, RuntimeContext, new_run_id, normalize_params, raise_for_support, resolve_runtime_context
from .errors import RuntimeFailure, SetupFailure
from .export import DecompositionArtifacts, export_bundle
from .platform_support import build_runtime_env, subprocess_command
from .validate import ValidatedMeshRequest, validate_inputs

MODEL_WEIGHT_RELATIVE_PATH = Path("p3sam") / "p3sam.safetensors"
UPSTREAM_RUNTIME_ROOT = Path(".upstream") / "hunyuan3d-part"
UPSTREAM_ENTRYPOINT = Path("P3-SAM") / "demo" / "auto_mask.py"
UPSTREAM_REQUIRED_PATHS = (
    Path("P3-SAM") / "model.py",
    Path("P3-SAM") / "demo" / "auto_mask.py",
    Path("XPart") / "partgen" / "models",
)
DEFAULT_PROMPT_BATCH_SIZE = 32
SONATA_CACHE_ENV_VAR = "P3SAM_SONATA_CACHE"
IMPORT_SMOKE_MARKER = "__P3SAM_IMPORT_SMOKE__="
PRIMARY_ARTIFACT_BASENAME = "auto_mask_mesh_final"
P3_SAM_STDOUT_LOG_NAME = "p3_sam_stdout.log"
P3_SAM_STDERR_LOG_NAME = "p3_sam_stderr.log"
P3_SAM_FAILURE_JSON_NAME = "p3_sam_failure.json"
P3_SAM_EXPECTED_OUTPUTS = {
    "primary_glb": f"{PRIMARY_ARTIFACT_BASENAME}.glb",
    "primary_ply": f"{PRIMARY_ARTIFACT_BASENAME}.ply",
    "aabb_glb": f"{PRIMARY_ARTIFACT_BASENAME}_aabb.glb",
    "aabb_npy": f"{PRIMARY_ARTIFACT_BASENAME}_aabb.npy",
    "face_ids_npy": f"{PRIMARY_ARTIFACT_BASENAME}_face_ids.npy",
}


@dataclass(frozen=True)
class ExecutionPlan:
    mesh: ValidatedMeshRequest
    params: NormalizedParams
    context: RuntimeContext
    run_id: str
    image_evidence: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        support = self.context.support.to_dict()
        support.pop("diagnostics", None)
        support.pop("resource_limits", None)
        payload: dict[str, object] = {
            "mesh": str(self.mesh.mesh_path),
            "mesh_format": self.mesh.mesh_format,
            "params": self.params.to_dict(),
            "run_id": self.run_id,
            "host": self.context.host_facts.to_dict(),
            "support": support,
        }
        if self.image_evidence is not None:
            payload["image_evidence"] = dict(self.image_evidence)
        return payload


@dataclass(frozen=True)
class AdapterPaths:
    managed_python: Path
    runtime_root: Path
    entrypoint: Path
    weights: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "managed_python": str(self.managed_python),
            "runtime_root": str(self.runtime_root),
            "entrypoint": str(self.entrypoint),
            "weights": str(self.weights),
        }


@dataclass(frozen=True)
class AdapterReadiness:
    ready: bool
    status: str
    paths: AdapterPaths | None
    components: dict[str, dict[str, object]]
    message: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ready": self.ready,
            "status": self.status,
            "components": self.components,
            "message": self.message,
        }
        if self.paths is not None:
            payload["paths"] = self.paths.to_dict()
        return payload


RuntimeAdapter = Callable[[ExecutionPlan], DecompositionArtifacts]
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_execution_plan(
    inputs: dict[str, object],
    params: dict[str, object] | None = None,
    *,
    runtime_context: RuntimeContext | None = None,
    project_root: Path | None = None,
    image_evidence: Mapping[str, object] | None = None,
) -> ExecutionPlan:
    mesh = validate_inputs(inputs)
    normalized = normalize_params(params)
    context = runtime_context or resolve_runtime_context(project_root=project_root)
    raise_for_support(context.support)
    return ExecutionPlan(mesh=mesh, params=normalized, context=context, run_id=new_run_id(), image_evidence=image_evidence)


def runtime_source_root(project_root: Path) -> Path:
    return project_root / UPSTREAM_RUNTIME_ROOT


def runtime_entrypoint(project_root: Path) -> Path:
    return runtime_source_root(project_root) / UPSTREAM_ENTRYPOINT


def sonata_cache_root(project_root: Path) -> Path:
    return project_root / ".cache" / "sonata"


def resolve_weight_path(model_root: Path | None) -> Path:
    if model_root is None:
        raise SetupFailure(
            "Model root is required to resolve the managed P3-SAM checkpoint.",
            code="missing_model_root",
        )

    root = model_root.resolve()
    direct_candidate = root / MODEL_WEIGHT_RELATIVE_PATH
    already_nested_candidate = root / MODEL_WEIGHT_RELATIVE_PATH.name
    if root.name == MODEL_WEIGHT_RELATIVE_PATH.parent.name and already_nested_candidate.is_file():
        return already_nested_candidate
    if direct_candidate.is_file():
        return direct_candidate

    candidates = sorted(root.glob(f"**/{MODEL_WEIGHT_RELATIVE_PATH.as_posix()}"))
    if candidates:
        return candidates[0]

    if root.name == MODEL_WEIGHT_RELATIVE_PATH.parent.name:
        return root / MODEL_WEIGHT_RELATIVE_PATH.name
    return direct_candidate


def _extract_missing_modules(*chunks: str | None) -> list[str]:
    missing: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        for match in re.finditer(r"No module named ['\"]([^'\"]+)['\"]", chunk):
            module_name = match.group(1)
            if module_name not in missing:
                missing.append(module_name)
    return missing


def _extract_native_blockers(*chunks: str | None) -> list[str]:
    blockers: list[str] = []
    for candidate in ("spconv", "torch_scatter"):
        if any(chunk and candidate in chunk for chunk in chunks):
            blockers.append(candidate)
    return blockers


def verify_upstream_import_smoke(
    *,
    project_root: Path,
    managed_python: Path,
) -> dict[str, object]:
    runtime_root = runtime_source_root(project_root)
    model_path = runtime_root / "P3-SAM" / "model.py"
    cache_root = sonata_cache_root(project_root)
    base_payload = {
        "path": str(model_path),
        "managed_python": str(managed_python),
        "cache_root": str(cache_root),
    }
    if not managed_python.exists():
        return {
            **base_payload,
            "ready": False,
            "status": "missing",
            "reason": "managed_python_missing",
        }
    if not model_path.is_file():
        return {
            **base_payload,
            "ready": False,
            "status": "missing",
            "reason": "upstream_model_missing",
        }

    script = "\n".join(
        [
            "import importlib.util",
            "import json",
            "import pathlib",
            "import sys",
            "import traceback",
            "try:",
            "    import torch  # import first so Windows DLL paths are initialized before upstream imports",
            "except Exception:",
            "    pass",
            f"model_path = pathlib.Path({str(model_path)!r})",
            "payload = {'ready': False, 'python_exe': sys.executable, 'module': str(model_path)}",
            "try:",
            "    spec = importlib.util.spec_from_file_location('p3_sam_import_smoke', str(model_path))",
            "    if spec is None or spec.loader is None:",
            "        raise RuntimeError('Failed to create import spec for upstream P3-SAM model.')",
            "    module = importlib.util.module_from_spec(spec)",
            "    spec.loader.exec_module(module)",
            "except Exception as exc:",
            "    payload.update({'error_type': type(exc).__name__, 'error': str(exc), 'traceback_tail': traceback.format_exc().splitlines()[-20:]})",
            f"    print({IMPORT_SMOKE_MARKER!r} + json.dumps(payload, sort_keys=True))",
            "    raise SystemExit(1)",
            "payload['ready'] = True",
            f"print({IMPORT_SMOKE_MARKER!r} + json.dumps(payload, sort_keys=True))",
        ]
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env[SONATA_CACHE_ENV_VAR] = str(cache_root)
    result = subprocess.run(
        subprocess_command(managed_python, "-c", script),
        check=False,
        capture_output=True,
        text=True,
        cwd=str(runtime_root / "P3-SAM"),
        env=env,
    )
    marker_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(IMPORT_SMOKE_MARKER)),
        None,
    )
    payload: dict[str, object] = {}
    if marker_line is not None:
        try:
            payload = json.loads(marker_line[len(IMPORT_SMOKE_MARKER) :])
        except json.JSONDecodeError:
            payload = {}
    combined_stdout = result.stdout.strip()
    combined_stderr = result.stderr.strip()
    missing_modules = _extract_missing_modules(combined_stdout, combined_stderr, str(payload.get("error", "")))
    native_blockers = _extract_native_blockers(combined_stdout, combined_stderr, str(payload.get("error", "")))
    ready = result.returncode == 0 and bool(payload.get("ready", False))
    return {
        **base_payload,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "returncode": result.returncode,
        "python_exe": payload.get("python_exe") or str(managed_python),
        "stdout": combined_stdout[-4000:],
        "stderr": combined_stderr[-4000:],
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "traceback_tail": payload.get("traceback_tail") or [],
        "missing_modules": missing_modules,
        "native_blockers": native_blockers,
    }


def build_adapter_readiness(
    *,
    project_root: Path,
    managed_python: Path,
    model_root: Path | None,
) -> AdapterReadiness:
    runtime_root = runtime_source_root(project_root)
    entrypoint = runtime_entrypoint(project_root)
    weights = resolve_weight_path(model_root) if model_root is not None else None
    components: dict[str, dict[str, object]] = {
        "managed_python": {
            "ready": managed_python.exists(),
            "path": str(managed_python),
            "status": "ready" if managed_python.exists() else "missing",
        },
        "runtime_source": {
            "ready": all((runtime_root / relative_path).exists() for relative_path in UPSTREAM_REQUIRED_PATHS),
            "path": str(runtime_root),
            "status": "ready" if all((runtime_root / relative_path).exists() for relative_path in UPSTREAM_REQUIRED_PATHS) else "missing",
            "required_paths": [str(relative_path) for relative_path in UPSTREAM_REQUIRED_PATHS],
        },
        "entrypoint": {
            "ready": entrypoint.is_file(),
            "path": str(entrypoint),
            "status": "ready" if entrypoint.is_file() else "missing",
        },
        "weights": {
            "ready": bool(weights and weights.is_file()),
            "path": str(weights) if weights is not None else None,
            "status": "ready" if weights is not None and weights.is_file() else "missing",
        },
    }
    components["import_smoke"] = verify_upstream_import_smoke(
        project_root=project_root,
        managed_python=managed_python,
    )
    ready = all(bool(section["ready"]) for section in components.values())
    paths = None
    if weights is not None:
        paths = AdapterPaths(
            managed_python=managed_python,
            runtime_root=runtime_root,
            entrypoint=entrypoint,
            weights=weights,
        )
    message = (
        "P3-SAM adapter prerequisites are ready."
        if ready
        else "P3-SAM adapter is fail-closed until managed python, upstream runtime source, entrypoint, import smoke, and weights are all ready."
    )
    return AdapterReadiness(
        ready=ready,
        status="ready" if ready else "blocked",
        paths=paths,
        components=components,
        message=message,
    )


def _require_adapter_paths(readiness: AdapterReadiness) -> AdapterPaths:
    if readiness.ready and readiness.paths is not None:
        return readiness.paths
    raise SetupFailure(
        readiness.message,
        code="runtime_adapter_unavailable",
        details=readiness.to_dict(),
    )


def build_subprocess_command(
    *,
    managed_python: Path,
    entrypoint: Path,
    weights: Path,
    mesh_path: Path,
    output_dir: Path,
    params: NormalizedParams,
) -> list[str]:
    command = subprocess_command(
        managed_python,
        entrypoint,
        "--ckpt_path",
        weights,
        "--mesh_path",
        mesh_path,
        "--output_path",
        output_dir,
        "--seed",
        str(params.seed if params.seed is not None else 42),
        "--prompt_bs",
        str(DEFAULT_PROMPT_BATCH_SIZE),
        "--show_info",
        "0",
        "--show_time_info",
        "0",
        "--save_mid_res",
        "1",
        "--post_process",
        "0",
        "--parallel",
        "0",
    )
    return command


def _load_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_face_ids(face_ids_path: Path) -> list[int]:
    if not face_ids_path.is_file():
        return []
    import numpy as np

    face_ids = np.load(face_ids_path)
    return [int(item) for item in face_ids.tolist()]


def _load_raw_aabbs(aabb_path: Path) -> list[object]:
    if not aabb_path.is_file():
        return []
    import numpy as np

    aabb = np.load(aabb_path)
    if aabb.ndim == 4 and aabb.shape[0] == 1:
        aabb = aabb[0]
    return aabb.tolist()


def _load_bboxes(aabb_path: Path) -> dict[str, object]:
    aabb = _load_raw_aabbs(aabb_path)
    parts: list[dict[str, object]] = []
    for index, part in enumerate(aabb):
        if len(part) != 2:
            continue
        min_xyz, max_xyz = part
        parts.append({"part_id": f"part-{index}", "min": min_xyz, "max": max_xyz})
    return {"parts": parts}


def _raw_part_ids(face_ids: list[int]) -> list[int]:
    return sorted({part_id for part_id in face_ids if part_id >= 0})


def _face_counts_by_part(face_ids: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for part_id in face_ids:
        if part_id < 0:
            continue
        counts[part_id] = counts.get(part_id, 0) + 1
    return counts


def _select_effective_parts(face_ids: list[int], max_parts: int) -> list[dict[str, int]]:
    counts = _face_counts_by_part(face_ids)
    selected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_parts]
    return [
        {"part_id": index, "raw_part_id": raw_part_id, "face_count": face_count}
        for index, (raw_part_id, face_count) in enumerate(selected)
    ]


def _reindex_effective_face_ids(face_ids: list[int], effective_parts: list[dict[str, int]]) -> list[int]:
    raw_to_effective = {int(part["raw_part_id"]): int(part["part_id"]) for part in effective_parts}
    return [raw_to_effective.get(part_id, -1) if part_id >= 0 else part_id for part_id in face_ids]


def _effective_bboxes(aabb_path: Path, raw_part_ids: list[int], effective_parts: list[dict[str, int]]) -> tuple[dict[str, object], Path | None]:
    raw_aabbs = _load_raw_aabbs(aabb_path)
    aabb_by_raw_part = {
        raw_part_id: raw_aabbs[index]
        for index, raw_part_id in enumerate(raw_part_ids)
        if index < len(raw_aabbs)
    }
    filtered_aabbs: list[object] = []
    parts: list[dict[str, object]] = []
    for part in effective_parts:
        raw_part_id = int(part["raw_part_id"])
        raw_aabb = aabb_by_raw_part.get(raw_part_id)
        if not raw_aabb or len(raw_aabb) != 2:
            continue
        min_xyz, max_xyz = raw_aabb
        filtered_aabbs.append(raw_aabb)
        parts.append(
            {
                "part_id": f"part-{part['part_id']}",
                "raw_part_id": raw_part_id,
                "min": min_xyz,
                "max": max_xyz,
            }
        )
    if not aabb_path.is_file():
        return {"source": "p3-sam", "parts": parts}, None
    import numpy as np

    effective_aabb_path = aabb_path.with_name(f"{aabb_path.stem}_effective.npy")
    np.save(effective_aabb_path, np.array(filtered_aabbs, dtype=float))
    return {"source": "p3-sam", "parts": parts}, effective_aabb_path


def _resolve_expected_output_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / value for key, value in P3_SAM_EXPECTED_OUTPUTS.items()}


def _persist_failure_diagnostics(
    *,
    output_dir: Path,
    command: list[str],
    cwd: str,
    result: subprocess.CompletedProcess[str],
) -> dict[str, str]:
    stdout_path = output_dir / P3_SAM_STDOUT_LOG_NAME
    stderr_path = output_dir / P3_SAM_STDERR_LOG_NAME
    failure_path = output_dir / P3_SAM_FAILURE_JSON_NAME
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    failure_path.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": cwd,
                "returncode": result.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "failure": str(failure_path),
    }


def collect_artifacts(output_dir: Path, *, export_format: str, params: NormalizedParams | None = None) -> DecompositionArtifacts:
    expected_paths = _resolve_expected_output_paths(output_dir)
    missing = [name for name, path in expected_paths.items() if not path.is_file()]
    primary_mesh = expected_paths[f"primary_{export_format}"]
    if missing:
        raise RuntimeFailure(
            "Upstream P3-SAM run completed without producing the expected save_mid_res artifacts.",
            code="missing_expected_artifacts",
            details={
                "missing": missing,
                "expected": {name: str(path) for name, path in expected_paths.items()},
                "output_dir": str(output_dir),
            },
        )

    raw_face_ids = _load_face_ids(expected_paths["face_ids_npy"])
    raw_part_ids = _raw_part_ids(raw_face_ids)
    max_parts = params.max_parts if params and params.max_parts is not None else DEFAULT_MAX_PARTS
    effective_parts = _select_effective_parts(raw_face_ids, max_parts)
    effective_face_ids = _reindex_effective_face_ids(raw_face_ids, effective_parts)
    bboxes, effective_aabb_path = _effective_bboxes(expected_paths["aabb_npy"], raw_part_ids, effective_parts)
    segmentation = {
        "face_ids": effective_face_ids,
        "source": "p3-sam",
        "raw_part_count": len(raw_part_ids),
        "effective_part_count": len(effective_parts),
        "max_parts": int(max_parts),
        "selection_policy": "face_count_desc",
        "semantic": False,
        "effective_parts": effective_parts,
    }
    completion = _load_optional_json(output_dir / "completion.json") or {
        "status": "completed",
        "adapter": "p3-sam-upstream-subprocess",
    }
    return DecompositionArtifacts(
        primary_mesh_path=primary_mesh,
        parts=(),
        segmentation=segmentation,
        bboxes=bboxes,
        completion=completion,
        metadata={
            "adapter": "p3-sam-upstream-subprocess",
            "output_dir": str(output_dir),
            "raw_outputs": {name: str(path) for name, path in expected_paths.items()},
            "aabb_path": str(expected_paths["aabb_npy"]),
            "effective_aabb_path": str(effective_aabb_path) if effective_aabb_path is not None else None,
        },
    )


def run_upstream_p3_sam(
    plan: ExecutionPlan,
    *,
    project_root: Path,
    managed_python: Path,
    model_root: Path | None,
    output_dir: Path,
    runner: SubprocessRunner = subprocess.run,
) -> DecompositionArtifacts:
    if plan.params.export_format != "glb":
        raise SetupFailure(
            "The real upstream P3-SAM adapter currently supports GLB primary export only.",
            code="unsupported_runtime_export_format",
            details={"requested": plan.params.export_format, "supported": ["glb"]},
        )

    readiness = build_adapter_readiness(
        project_root=project_root,
        managed_python=managed_python,
        model_root=model_root,
    )
    paths = _require_adapter_paths(readiness)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_subprocess_command(
        managed_python=paths.managed_python,
        entrypoint=paths.entrypoint,
        weights=paths.weights,
        mesh_path=plan.mesh.mesh_path,
        output_dir=output_dir,
        params=plan.params,
    )
    env = build_runtime_env(
        os.environ,
        pythonpath=(runtime_source_root(project_root),),
        extra={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", SONATA_CACHE_ENV_VAR: str(sonata_cache_root(project_root))},
    )
    result = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(paths.entrypoint.parent),
        env=env,
    )
    if result.returncode != 0:
        diagnostics = _persist_failure_diagnostics(
            output_dir=output_dir,
            command=command,
            cwd=str(paths.entrypoint.parent),
            result=result,
        )
        raise RuntimeFailure(
            "Upstream P3-SAM subprocess failed. See persisted diagnostics in the output directory.",
            code="p3_sam_subprocess_failed",
            details={
                "command": command,
                "cwd": str(paths.entrypoint.parent),
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "diagnostics": diagnostics,
            },
        )
    return collect_artifacts(output_dir, export_format=plan.params.export_format, params=plan.params)


def decompose_mesh(
    inputs: dict[str, object],
    params: dict[str, object] | None = None,
    *,
    output_dir: str | Path,
    image_evidence: Mapping[str, object] | None = None,
    runtime_adapter: RuntimeAdapter | None = None,
    runtime_context: RuntimeContext | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    plan = build_execution_plan(
        inputs,
        params,
        runtime_context=runtime_context,
        project_root=project_root,
        image_evidence=image_evidence,
    )
    if runtime_adapter is None:
        raise SetupFailure(
            "Live execution requires a configured P3-SAM runtime adapter.",
            code="missing_runtime_adapter",
            details={"pipeline_stage": plan.params.pipeline_stage},
        )
    artifacts = runtime_adapter(plan)
    if not isinstance(artifacts, DecompositionArtifacts):
        raise RuntimeFailure(
            "Runtime adapter returned an invalid artifact payload.",
            code="invalid_runtime_adapter_payload",
        )
    return export_bundle(
        artifacts=artifacts,
        output_dir=Path(output_dir),
        params=plan.params,
        context=plan.context,
        run_id=plan.run_id,
    )
