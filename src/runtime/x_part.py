"""Real X-Part runtime adapter plumbing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping

from .config import NormalizedParams, resolve_x_part_resource_limits
from .errors import RuntimeFailure, SetupFailure
from .export import DecompositionArtifacts, PartArtifact
from .platform_support import build_runtime_env, memory_guard_status, subprocess_command
from .p3_sam import ExecutionPlan, SubprocessRunner, runtime_source_root

X_PART_PIPELINE_RELATIVE_PATH = Path("XPart") / "partgen" / "partformer_pipeline.py"
X_PART_IMPORT_SMOKE_MARKER = "__XPART_IMPORT_SMOKE__="
X_PART_PRIMARY_NAME = "x_part_primary.glb"
X_PART_STDOUT_LOG_NAME = "x_part_stdout.log"
X_PART_STDERR_LOG_NAME = "x_part_stderr.log"
X_PART_FAILURE_JSON_NAME = "x_part_failure.json"
X_PART_RESOURCE_LIMITS_JSON_NAME = "x_part_resource_limits.json"
X_PART_MODEL_RELATIVE_PATH = Path("model") / "model.safetensors"
X_PART_MARKER_DIRS = ("conditioner", "shapevae", "scheduler")
X_PART_CONFIG_MARKERS = ("config.json",)
X_PART_TIMEOUT_ENV_VAR = "HUNYUAN3D_PART_XPART_TIMEOUT_SECONDS"
X_PART_MEMORY_GUARD_ENV_VAR = "HUNYUAN3D_PART_MIN_AVAILABLE_MEMORY_GIB"
DEFAULT_X_PART_TIMEOUT_SECONDS = 30 * 60
DEFAULT_X_PART_MIN_AVAILABLE_MEMORY_GIB = 16.0
X_PART_MEMORY_GUARD_POLL_SECONDS = 1.0
X_PART_DIAGNOSTIC_TAIL_CHARS = 4000
X_PART_DIAGNOSTIC_TAIL_LINES = 80
WINDOWS_LOW_VRAM_X_PART_LIMITS = {
    "num_inference_steps": 6,
    "octree_resolution": 192,
    "num_chunks": 4096,
    "surface_point_count": 1024,
    "requested_bbox_point_count": 1024,
    "total_bbox_point_budget": 4096,
    "min_bbox_point_count": 512,
    "effective_max_parts_cap": 3,
    "torch_dtype": "float16",
}


def runtime_pipeline_path(project_root: Path) -> Path:
    return runtime_source_root(project_root) / X_PART_PIPELINE_RELATIVE_PATH


def x_part_import_root(pipeline_path: Path) -> Path:
    return pipeline_path.parents[1]


def _inspect_bundle_candidate(candidate: Path) -> dict[str, object]:
    resolved = candidate.resolve()
    is_dir = resolved.is_dir()
    model_path = resolved / X_PART_MODEL_RELATIVE_PATH
    marker_dirs_present = [name for name in X_PART_MARKER_DIRS if (resolved / name).is_dir()] if is_dir else []
    config_markers_present = [name for name in X_PART_CONFIG_MARKERS if (resolved / name).is_file()] if is_dir else []
    p3_sam_weight = resolved / "p3sam.safetensors"
    valid = bool(model_path.is_file() and (marker_dirs_present or config_markers_present))
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": is_dir,
        "model_path": str(model_path),
        "model_present": model_path.is_file(),
        "marker_dirs_present": marker_dirs_present,
        "config_markers_present": config_markers_present,
        "p3_sam_weight_present": p3_sam_weight.is_file(),
        "valid": valid,
    }


def resolve_bundle_root(model_root: Path | None) -> Path:
    if model_root is None:
        raise SetupFailure(
            "Model root is required to resolve the X-Part bundle root.",
            code="missing_model_root",
        )
    root = model_root.resolve()
    candidates = [
        root,
        root / "p3sam",
        root / "p3sam" / "p3sam",
        root / "model",
        root / "x-part",
        root / "xpart",
        root / "XPart",
    ]
    candidate_statuses = [_inspect_bundle_candidate(candidate) for candidate in candidates]
    preferred_order = [root / "p3sam", root, root / "p3sam" / "p3sam", root / "model", root / "x-part", root / "xpart", root / "XPart"]
    for candidate in preferred_order:
        status = next((item for item in candidate_statuses if item["path"] == str(candidate.resolve())), None)
        if status and status["valid"]:
            return candidate
    raise SetupFailure(
        "Unable to resolve an existing X-Part bundle root from the managed model cache.",
        code="missing_x_part_bundle",
        details={
            "model_root": str(root),
            "required_model_path": X_PART_MODEL_RELATIVE_PATH.as_posix(),
            "required_marker_dirs": list(X_PART_MARKER_DIRS),
            "config_markers": list(X_PART_CONFIG_MARKERS),
            "candidates": candidate_statuses,
        },
    )


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


def verify_x_part_import_smoke(*, project_root: Path, managed_python: Path, model_root: Path | None) -> dict[str, object]:
    pipeline_path = runtime_pipeline_path(project_root)
    import_root = x_part_import_root(pipeline_path)
    bundle_root = resolve_bundle_root(model_root)
    bundle_validation = _inspect_bundle_candidate(bundle_root)
    payload = {
        "managed_python": str(managed_python),
        "pipeline_path": str(pipeline_path),
        "import_root": str(import_root),
        "bundle_root": str(bundle_root),
        "bundle_validation": bundle_validation,
    }
    if not managed_python.exists():
        return {**payload, "ready": False, "status": "missing", "reason": "managed_python_missing"}
    if not pipeline_path.is_file():
        return {**payload, "ready": False, "status": "missing", "reason": "pipeline_missing"}
    script = "\n".join(
        [
            "import json",
            "import sys",
            "from pathlib import Path",
            "try:",
            "    import torch  # import first so Windows DLL paths are initialized before upstream imports",
            "except Exception:",
            "    pass",
            f"pipeline_path = Path({str(pipeline_path)!r})",
            f"import_root = Path({str(import_root)!r})",
            "sys.path.insert(0, str(import_root))",
            "payload = {'ready': False, 'python_exe': sys.executable}",
            "payload['import_root'] = str(import_root)",
            "try:",
            "    from partgen.partformer_pipeline import PartFormerPipeline",
            "    payload['class_name'] = PartFormerPipeline.__name__",
            "except Exception as exc:",
            "    payload['error_type'] = type(exc).__name__",
            "    payload['error'] = str(exc)",
            f"    print({X_PART_IMPORT_SMOKE_MARKER!r} + json.dumps(payload, sort_keys=True))",
            "    raise SystemExit(1)",
            "payload['ready'] = True",
            f"print({X_PART_IMPORT_SMOKE_MARKER!r} + json.dumps(payload, sort_keys=True))",
        ]
    )
    result = subprocess.run(
        subprocess_command(managed_python, "-c", script),
        check=False,
        capture_output=True,
        text=True,
        cwd=str(pipeline_path.parent),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
    )
    marker_line = next(
        (line for line in reversed(result.stdout.splitlines()) if line.startswith(X_PART_IMPORT_SMOKE_MARKER)),
        None,
    )
    smoke_payload: dict[str, object] = {}
    if marker_line is not None:
        try:
            smoke_payload = json.loads(marker_line[len(X_PART_IMPORT_SMOKE_MARKER) :])
        except json.JSONDecodeError:
            smoke_payload = {}
    missing_modules = _extract_missing_modules(result.stdout, result.stderr, str(smoke_payload.get("error", "")))
    ready = result.returncode == 0 and bool(smoke_payload.get("ready", False))
    return {
        **payload,
        "ready": ready,
        "status": "ready" if ready else "blocked",
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "class_name": smoke_payload.get("class_name"),
        "error_type": smoke_payload.get("error_type"),
        "error": smoke_payload.get("error"),
        "missing_modules": missing_modules,
    }


def _part_paths(parts_dir: Path) -> Iterable[Path]:
    for path in sorted(parts_dir.glob("*.glb")):
        if path.is_file():
            yield path


def resolve_adapter_x_part_resource_limits(params: NormalizedParams, *, host_os_name: str | None = None) -> dict[str, object]:
    """Return X-Part limits adapted for the concrete adapter host.

    The global config resolver remains the Linux/default contract.  Windows gets
    an adapter-only low-VRAM policy because the managed X-Part subprocess has a
    weaker memory guard there when ``/proc`` is unavailable and ``psutil`` is not
    installed.
    """

    limits = dict(resolve_x_part_resource_limits(params))
    requested_quality_preset = str(limits.get("quality_preset", params.quality_preset or "balanced"))
    limits.setdefault("requested_quality_preset", requested_quality_preset)
    limits.setdefault("effective_quality_preset", requested_quality_preset)
    limits.setdefault("platform_policy", "default")
    if (host_os_name or "").lower() != "windows":
        return limits

    requested_effective_max_parts = int(limits["effective_max_parts"])
    windows_effective_max_parts_cap = int(WINDOWS_LOW_VRAM_X_PART_LIMITS["effective_max_parts_cap"])
    effective_max_parts = min(requested_effective_max_parts, windows_effective_max_parts_cap)
    requested_bbox_point_count = min(int(limits.get("requested_bbox_point_count", limits.get("bbox_point_count", 4096))), int(WINDOWS_LOW_VRAM_X_PART_LIMITS["requested_bbox_point_count"]))
    total_bbox_point_budget = int(WINDOWS_LOW_VRAM_X_PART_LIMITS["total_bbox_point_budget"])
    min_bbox_point_count = int(WINDOWS_LOW_VRAM_X_PART_LIMITS["min_bbox_point_count"])
    adaptive_bbox_point_count = min(
        requested_bbox_point_count,
        max(min_bbox_point_count, total_bbox_point_budget // max(1, effective_max_parts)),
    )
    limits.update(
        {
            "platform_policy": "windows_low_vram_safe",
            "requested_quality_preset": requested_quality_preset,
            "effective_quality_preset": "fast",
            "quality_preset": requested_quality_preset,
            "requested_effective_max_parts": requested_effective_max_parts,
            "windows_effective_max_parts_cap": windows_effective_max_parts_cap,
            "effective_max_parts": effective_max_parts,
            "num_inference_steps": int(WINDOWS_LOW_VRAM_X_PART_LIMITS["num_inference_steps"]),
            "octree_resolution": int(WINDOWS_LOW_VRAM_X_PART_LIMITS["octree_resolution"]),
            "num_chunks": min(int(limits.get("num_chunks", WINDOWS_LOW_VRAM_X_PART_LIMITS["num_chunks"])), int(WINDOWS_LOW_VRAM_X_PART_LIMITS["num_chunks"])),
            "surface_point_count": min(int(limits.get("surface_point_count", WINDOWS_LOW_VRAM_X_PART_LIMITS["surface_point_count"])), int(WINDOWS_LOW_VRAM_X_PART_LIMITS["surface_point_count"])),
            "requested_bbox_point_count": requested_bbox_point_count,
            "bbox_point_count": adaptive_bbox_point_count,
            "total_bbox_point_budget": total_bbox_point_budget,
            "min_bbox_point_count": min_bbox_point_count,
            "point_budget_policy": "windows_low_vram_cap_total_bbox_points_by_effective_max_parts",
            "torch_dtype": "float16",
        }
    )
    return limits


def collect_artifacts(output_dir: Path) -> DecompositionArtifacts:
    primary_mesh = output_dir / X_PART_PRIMARY_NAME
    segmentation_path = output_dir / "segmentation.json"
    bboxes_path = output_dir / "bboxes.json"
    completion_path = output_dir / "completion.json"
    if not primary_mesh.is_file():
        raise RuntimeFailure(
            "X-Part subprocess completed without producing the expected primary GLB.",
            code="missing_x_part_primary_artifact",
            details={"expected": str(primary_mesh), "output_dir": str(output_dir)},
        )
    if not segmentation_path.is_file() or not bboxes_path.is_file() or not completion_path.is_file():
        raise RuntimeFailure(
            "X-Part subprocess completed without producing the expected sidecar artifacts.",
            code="missing_x_part_sidecars",
            details={
                "segmentation": str(segmentation_path),
                "bboxes": str(bboxes_path),
                "completion": str(completion_path),
                "output_dir": str(output_dir),
            },
        )
    segmentation = json.loads(segmentation_path.read_text(encoding="utf-8"))
    bboxes = json.loads(bboxes_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    bbox_index = {item["part_id"]: item for item in bboxes.get("parts", []) if isinstance(item, dict) and item.get("part_id")}
    parts = tuple(
        PartArtifact(
            part_id=path.stem,
            mesh_path=path,
            bbox=bbox_index.get(path.stem, {}),
            label=path.stem,
            metadata={"source": "x-part"},
        )
        for path in _part_paths(output_dir / "parts")
    )
    return DecompositionArtifacts(
        primary_mesh_path=primary_mesh,
        parts=parts,
        segmentation=segmentation,
        bboxes=bboxes,
        completion=completion,
        metadata={"adapter": "x-part-upstream-subprocess", "output_dir": str(output_dir)},
    )


def _tail_text(value: str | None, *, max_chars: int = X_PART_DIAGNOSTIC_TAIL_CHARS, max_lines: int = X_PART_DIAGNOSTIC_TAIL_LINES) -> str:
    if not value:
        return ""
    lines = value.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if value.endswith("\n") and tail:
        tail += "\n"
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _existing_artifacts(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return []
    return sorted(str(path) for path in output_dir.iterdir() if path.is_file())


def _load_resource_limits_payload(resource_limits_path: Path | None) -> dict[str, object]:
    if resource_limits_path is None or not resource_limits_path.is_file():
        return {}
    try:
        return json.loads(resource_limits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"resource_limits_load_error": str(exc)}


def _resource_limits_summary(resource_limits: Mapping[str, object] | None) -> dict[str, object]:
    if not resource_limits:
        return {}
    keys = (
        "requested_effective_max_parts",
        "windows_effective_max_parts_cap",
        "effective_max_parts",
        "max_parts_hard_cap",
        "quality_preset",
        "requested_quality_preset",
        "effective_quality_preset",
        "platform_policy",
        "num_inference_steps",
        "octree_resolution",
        "num_chunks",
        "surface_point_count",
        "bbox_point_count",
        "requested_bbox_point_count",
        "total_bbox_point_budget",
        "min_bbox_point_count",
        "point_budget_policy",
        "torch_dtype",
        "resolved_torch_dtype",
    )
    return {key: resource_limits[key] for key in keys if key in resource_limits}


def _last_memory_event(resource_limits: Mapping[str, object] | None) -> dict[str, object] | None:
    trace = resource_limits.get("memory_trace") if resource_limits else None
    if isinstance(trace, list) and trace and isinstance(trace[-1], dict):
        return trace[-1]
    return None


def _format_resource_summary(summary: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in summary.items())


def _is_low_vram_budget_pressure(last_memory_event: Mapping[str, object] | None) -> bool:
    if not last_memory_event:
        return False
    for key in ("torch_cuda_allocated_gib", "torch_cuda_reserved_gib"):
        try:
            if float(last_memory_event.get(key, 0.0)) >= 7.75:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _format_failure_message(
    *,
    reason: str,
    diagnostics: Mapping[str, str],
    output_dir: Path,
    resource_limits_path: Path,
    resource_summary: Mapping[str, object],
    stdout_tail: str,
    stderr_tail: str,
    existing_artifacts: Iterable[str],
    returncode: int | None = None,
    timeout_seconds: int | None = None,
    last_memory_stage: object | None = None,
    last_memory_event: Mapping[str, object] | None = None,
) -> str:
    first_line = reason
    if timeout_seconds is not None:
        first_line = f"{first_line} after {timeout_seconds} seconds."
    elif returncode is not None:
        first_line = f"{first_line} with return code {returncode}."
    lines = [
        first_line,
        "Persisted diagnostics:",
        f"- failure_json: {diagnostics.get('failure')}",
        f"- stderr_log: {diagnostics.get('stderr')}",
        f"- stdout_log: {diagnostics.get('stdout')}",
        f"- output_dir: {output_dir}",
        f"- resource_limits_json: {resource_limits_path}",
    ]
    if resource_summary:
        lines.append(f"Resource summary: {_format_resource_summary(resource_summary)}")
    if last_memory_stage is not None:
        lines.append(f"Last memory stage: {last_memory_stage}")
    if last_memory_event:
        lines.append(f"Last memory event: {json.dumps(dict(last_memory_event), sort_keys=True)}")
        if _is_low_vram_budget_pressure(last_memory_event):
            lines.append("Hint: X-Part is over low-VRAM budget; reduce max_parts/quality or use P3-SAM stage.")
    artifact_list = list(existing_artifacts)
    if artifact_list:
        lines.extend(["Existing artifacts:", *[f"- {path}" for path in artifact_list]])
    if stderr_tail:
        lines.extend(["", "stderr tail:", stderr_tail.rstrip()])
    if stdout_tail:
        lines.extend(["", "stdout tail:", stdout_tail.rstrip()])
    return "\n".join(lines)


def _persist_failure_diagnostics(
    *,
    output_dir: Path,
    bundle_root: Path,
    mesh_path: Path,
    cwd: str,
    result: subprocess.CompletedProcess[str],
    aabb_path: Path | None,
    resource_limits_path: Path | None = None,
    timeout_seconds: int | None = None,
    timed_out: bool = False,
    memory_guard: dict[str, object] | None = None,
    adapter_resource_limits: Mapping[str, object] | None = None,
) -> dict[str, str]:
    stdout_path = output_dir / X_PART_STDOUT_LOG_NAME
    stderr_path = output_dir / X_PART_STDERR_LOG_NAME
    failure_path = output_dir / X_PART_FAILURE_JSON_NAME
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    stdout_tail = _tail_text(result.stdout)
    stderr_tail = _tail_text(result.stderr)
    resource_limits_payload = _load_resource_limits_payload(resource_limits_path)
    resource_limits_summary = _resource_limits_summary(resource_limits_payload or adapter_resource_limits)
    failure_path.write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "bundle_root": str(bundle_root),
                "mesh_path": str(mesh_path),
                "output_dir": str(output_dir),
                "aabb_path": str(aabb_path) if aabb_path is not None else None,
                "resource_limits_path": str(resource_limits_path) if resource_limits_path is not None else None,
                "timeout_seconds": timeout_seconds,
                "timed_out": timed_out,
                "memory_guard": memory_guard,
                "cwd": cwd,
                "script_summary": {
                    "entrypoint": str(X_PART_PIPELINE_RELATIVE_PATH),
                    "adapter": "x-part-upstream-subprocess",
                },
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "existing_artifacts": _existing_artifacts(output_dir),
                "resource_limits_summary": resource_limits_summary,
                "resource_limits_payload": resource_limits_payload,
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


def _normalize_timeout_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_x_part_timeout_seconds(env: dict[str, str] | None = None) -> int:
    current_env = env or os.environ
    raw_value = current_env.get(X_PART_TIMEOUT_ENV_VAR)
    if raw_value in (None, ""):
        return DEFAULT_X_PART_TIMEOUT_SECONDS
    try:
        timeout_seconds = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SetupFailure(
            "X-Part timeout override must be an integer number of seconds.",
            code="invalid_x_part_timeout",
            details={"env_var": X_PART_TIMEOUT_ENV_VAR, "observed": raw_value},
        ) from exc
    if timeout_seconds < 60:
        raise SetupFailure(
            "X-Part timeout override is too low for a managed runtime subprocess.",
            code="invalid_x_part_timeout",
            details={"env_var": X_PART_TIMEOUT_ENV_VAR, "observed": timeout_seconds, "minimum": 60},
        )
    return timeout_seconds


def _resolve_x_part_min_available_memory_gib(env: dict[str, str] | None = None) -> float:
    current_env = env or os.environ
    raw_value = current_env.get(X_PART_MEMORY_GUARD_ENV_VAR)
    if raw_value in (None, ""):
        return DEFAULT_X_PART_MIN_AVAILABLE_MEMORY_GIB
    try:
        threshold = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise SetupFailure(
            "X-Part memory guard override must be a number of GiB.",
            code="invalid_x_part_memory_guard",
            details={"env_var": X_PART_MEMORY_GUARD_ENV_VAR, "observed": raw_value},
        ) from exc
    if threshold < 0:
        raise SetupFailure(
            "X-Part memory guard override cannot be negative.",
            code="invalid_x_part_memory_guard",
            details={"env_var": X_PART_MEMORY_GUARD_ENV_VAR, "observed": threshold, "minimum": 0},
        )
    return threshold


def _read_mem_available_gib(meminfo_path: Path = Path("/proc/meminfo")) -> float | None:
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def _read_fallback_mem_available_gib() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def _run_with_memory_guard(
    command: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout_seconds: int,
    min_available_gib: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object] | None]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    started_at = time.monotonic()
    lowest_available_gib: float | None = None
    while process.poll() is None:
        available_gib = _read_mem_available_gib()
        guard_source = "/proc/meminfo"
        guard_status = "enforced"
        if available_gib is None:
            available_gib = _read_fallback_mem_available_gib()
            guard_source = "psutil.virtual_memory" if available_gib is not None else None
            guard_status = "fallback" if available_gib is not None else "degraded"
        if available_gib is not None:
            lowest_available_gib = available_gib if lowest_available_gib is None else min(lowest_available_gib, available_gib)
            if min_available_gib > 0 and available_gib < min_available_gib:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                message = (
                    f"\nX-Part memory guard terminated subprocess: MemAvailable={available_gib:.2f} GiB "
                    f"below threshold {min_available_gib:.2f} GiB.\n"
                )
                return (
                    subprocess.CompletedProcess(command, process.returncode if process.returncode is not None else -9, stdout, (stderr or "") + message),
                    {
                        "triggered": True,
                        "guard_status": guard_status,
                        "source": guard_source,
                        "enforced": True,
                        "available_gib": round(available_gib, 3),
                        "threshold_gib": round(min_available_gib, 3),
                        "lowest_available_gib": round(lowest_available_gib, 3) if lowest_available_gib is not None else None,
                    },
                )
        if time.monotonic() - started_at > timeout_seconds:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr)
        time.sleep(X_PART_MEMORY_GUARD_POLL_SECONDS)
    stdout, stderr = process.communicate()
    return (
        subprocess.CompletedProcess(command, process.returncode if process.returncode is not None else 0, stdout, stderr),
        {
            "triggered": False,
            "guard_status": memory_guard_status().get("guard_status") if lowest_available_gib is None else "enforced",
            "enforced": bool(lowest_available_gib is not None or memory_guard_status().get("enforced", False)),
            "threshold_gib": round(min_available_gib, 3),
            "lowest_available_gib": round(lowest_available_gib, 3) if lowest_available_gib is not None else None,
        },
    )


def _build_subprocess_script(
    *,
    runtime_root: Path,
    bundle_root: Path,
    mesh_path: Path,
    output_dir: Path,
    params: NormalizedParams,
    aabb_path: Path | None,
    host_os_name: str | None = None,
) -> str:
    x_part_root = x_part_import_root(runtime_root / X_PART_PIPELINE_RELATIVE_PATH)
    limits = resolve_adapter_x_part_resource_limits(params, host_os_name=host_os_name)
    seed = params.seed if params.seed is not None else 42
    return "\n".join(
        [
            "import json",
            "import time",
            "from pathlib import Path",
            "import sys",
            "import numpy as np",
            "import torch",
            "import trimesh",
            f"runtime_root = Path({str(runtime_root)!r})",
            f"x_part_root = Path({str(x_part_root)!r})",
            f"bundle_root = Path({str(bundle_root)!r})",
            f"mesh_path = Path({str(mesh_path)!r})",
            f"output_dir = Path({str(output_dir)!r})",
            f"resource_limits = {repr(limits)}",
            f"seed = {repr(seed)}",
            f"aabb_path = {repr(str(aabb_path) if aabb_path is not None else None)}",
            "sys.path.insert(0, str(x_part_root))",
            "output_dir.mkdir(parents=True, exist_ok=True)",
            "parts_dir = output_dir / 'parts'",
            "parts_dir.mkdir(parents=True, exist_ok=True)",
            "resource_limits_path = output_dir / 'x_part_resource_limits.json'",
            "diagnostic_state = {}",
            "memory_trace = []",
            "def write_resource_limits(**extra):",
            "    diagnostic_state.update(extra)",
            "    payload = dict(resource_limits)",
            "    payload.update(diagnostic_state)",
            "    resource_limits_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
            "    return payload",
            "def read_meminfo_kib():",
            "    info = {}",
            "    try:",
            "        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():",
            "            key, value = line.split(':', 1)",
            "            parts = value.strip().split()",
            "            if parts and parts[0].isdigit():",
            "                info[key] = int(parts[0])",
            "    except Exception as exc:",
            "        info['error'] = str(exc)",
            "    return info",
            "def read_self_status_kib():",
            "    info = {}",
            "    try:",
            "        for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines():",
            "            if line.startswith(('VmRSS:', 'VmSize:', 'VmHWM:')):",
            "                key, value = line.split(':', 1)",
            "                parts = value.strip().split()",
            "                if parts and parts[0].isdigit():",
            "                    info[key] = int(parts[0])",
            "    except Exception as exc:",
            "        info['error'] = str(exc)",
            "    return info",
            "def capture_memory(stage, **extra):",
            "    event = {'stage': stage, 'timestamp': time.time()}",
            "    meminfo = read_meminfo_kib()",
            "    status = read_self_status_kib()",
            "    if meminfo:",
            "        event['mem_available_gib'] = round(meminfo.get('MemAvailable', 0) / 1048576, 3)",
            "        event['mem_free_gib'] = round(meminfo.get('MemFree', 0) / 1048576, 3)",
            "        event['swap_free_gib'] = round(meminfo.get('SwapFree', 0) / 1048576, 3)",
            "    if status:",
            "        event['process_rss_gib'] = round(status.get('VmRSS', 0) / 1048576, 3)",
            "        event['process_hwm_gib'] = round(status.get('VmHWM', 0) / 1048576, 3)",
            "        event['process_vmsize_gib'] = round(status.get('VmSize', 0) / 1048576, 3)",
            "    if torch.cuda.is_available():",
            "        event['torch_cuda_allocated_gib'] = round(torch.cuda.memory_allocated() / 1073741824, 3)",
            "        event['torch_cuda_reserved_gib'] = round(torch.cuda.memory_reserved() / 1073741824, 3)",
            "    event.update(extra)",
            "    memory_trace.append(event)",
            "    resource_limits['memory_trace'] = memory_trace[-160:]",
            "    write_resource_limits(last_memory_stage=stage)",
            "capture_memory('script_initialized')",
            "if not torch.cuda.is_available():",
            "    raise RuntimeError('X-Part runtime requires CUDA, but torch.cuda.is_available() returned False.')",
            "device = torch.device('cuda', torch.cuda.current_device())",
            "torch.backends.cuda.matmul.allow_tf32 = True",
            "torch.backends.cudnn.allow_tf32 = True",
            "original_sdp_kernel = torch.backends.cuda.sdp_kernel",
            "def sm121_safe_sdp_kernel(*args, **kwargs):",
            "    return original_sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)",
            "torch.backends.cuda.sdp_kernel = sm121_safe_sdp_kernel",
            "resource_limits['sdp_kernel_policy'] = 'math_only_sm121_safe'",
            "requested_dtype = str(resource_limits.get('torch_dtype', 'bfloat16'))",
            "if requested_dtype == 'float32':",
            "    dtype = torch.float32",
            "elif requested_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():",
            "    dtype = torch.bfloat16",
            "else:",
            "    dtype = torch.float16",
            "capture_memory('cuda_ready', device=str(device), resolved_torch_dtype=str(dtype).replace('torch.', ''))",
            "capture_memory('before_aabb_load')",
            "aabb = np.load(aabb_path) if aabb_path else None",
            "raw_aabb_count = None",
            "effective_aabb_count = None",
            "if aabb is not None:",
            "    if aabb.ndim == 4 and aabb.shape[0] == 1:",
            "        aabb = aabb[0]",
            "    if aabb.ndim != 3 or tuple(aabb.shape[1:]) != (2, 3):",
            "        raise RuntimeError(f'Expected AABB shape [K, 2, 3], got {aabb.shape!r}.')",
            "    raw_aabb_count = int(aabb.shape[0])",
            "    aabb = aabb[: int(resource_limits['effective_max_parts'])]",
            "    effective_aabb_count = int(aabb.shape[0])",
            "capture_memory('after_aabb_trim', raw_aabb_count=raw_aabb_count, effective_aabb_count=effective_aabb_count)",
            "write_resource_limits(device=str(device), resolved_torch_dtype=str(dtype).replace('torch.', ''), raw_aabb_count=raw_aabb_count, effective_aabb_count=effective_aabb_count, external_aabb=bool(aabb_path), sdp_kernel_policy='math_only_sm121_safe')",
            "capture_memory('before_partgen_import')",
            "import partgen.partformer_pipeline as xpart_pipeline_module",
            "from partgen.partformer_pipeline import PartFormerPipeline",
            "capture_memory('after_partgen_import')",
            "original_load_surface_points = xpart_pipeline_module.load_surface_points",
            "def guarded_load_surface_points(rng, random_surface, sharp_surface, pc_size=81920, pc_sharpedge_size=0, *args, **kwargs):",
            "    effective_pc_size = min(int(pc_size), int(resource_limits['surface_point_count']))",
            "    capture_memory('before_load_surface_points', requested_pc_size=int(pc_size), effective_pc_size=effective_pc_size)",
            "    loaded = original_load_surface_points(rng, random_surface, sharp_surface, pc_size=effective_pc_size, pc_sharpedge_size=pc_sharpedge_size, *args, **kwargs)",
            "    capture_memory('after_load_surface_points', effective_pc_size=effective_pc_size)",
            "    return loaded",
            "xpart_pipeline_module.load_surface_points = guarded_load_surface_points",
            "original_sample_bbox_points_from_trimesh = xpart_pipeline_module.sample_bbox_points_from_trimesh",
            "def guarded_sample_bbox_points_from_trimesh(mesh, aabb, num_points=81920, *args, **kwargs):",
            "    effective_num_points = min(int(num_points), int(resource_limits['bbox_point_count']))",
            "    capture_memory('before_sample_bbox_points', requested_num_points=int(num_points), effective_num_points=effective_num_points)",
            "    sampled = original_sample_bbox_points_from_trimesh(mesh, aabb, num_points=effective_num_points, *args, **kwargs)",
            "    capture_memory('after_sample_bbox_points', effective_num_points=effective_num_points)",
            "    return sampled",
            "xpart_pipeline_module.sample_bbox_points_from_trimesh = guarded_sample_bbox_points_from_trimesh",
            "capture_memory('before_from_pretrained')",
            "pipeline = PartFormerPipeline.from_pretrained(str(bundle_root), device=device, dtype=dtype)",
            "capture_memory('after_from_pretrained')",
            "pipeline.to(device=device, dtype=dtype)",
            "capture_memory('after_pipeline_to')",
            "patched_encoder_pc_sizes = []",
            "for module_name, module in pipeline.conditioner.named_modules():",
            "    if hasattr(module, 'pc_size') and hasattr(module, 'pc_sharpedge_size'):",
            "        old_pc_size = int(getattr(module, 'pc_size'))",
            "        old_pc_sharpedge_size = int(getattr(module, 'pc_sharpedge_size'))",
            "        old_downsample_ratio = float(getattr(module, 'downsample_ratio', 1.0))",
            "        target_pc_size = int(resource_limits['bbox_point_count']) if 'geo_encoder' in module_name else int(resource_limits['surface_point_count'])",
            "        module.pc_size = min(old_pc_size, target_pc_size)",
            "        module.pc_sharpedge_size = 0",
            "        num_latents = max(1, int(getattr(module, 'num_latents', 1)))",
            "        max_downsample_ratio = max(1.0, float(module.pc_size) / float(num_latents))",
            "        module.downsample_ratio = min(old_downsample_ratio, max_downsample_ratio)",
            "        patched_encoder_pc_sizes.append({'module': module_name, 'old_pc_size': old_pc_size, 'new_pc_size': int(module.pc_size), 'old_pc_sharpedge_size': old_pc_sharpedge_size, 'new_pc_sharpedge_size': int(module.pc_sharpedge_size), 'old_downsample_ratio': old_downsample_ratio, 'new_downsample_ratio': float(module.downsample_ratio), 'num_latents': num_latents})",
            "if patched_encoder_pc_sizes:",
            "    write_resource_limits(device=str(device), resolved_torch_dtype=str(dtype).replace('torch.', ''), raw_aabb_count=raw_aabb_count, effective_aabb_count=effective_aabb_count, external_aabb=bool(aabb_path), patched_encoder_pc_sizes=patched_encoder_pc_sizes)",
            "capture_memory('after_encoder_patch', patched_encoder_count=len(patched_encoder_pc_sizes))",
            "if aabb is not None and getattr(pipeline, 'bbox_predictor', None) is not None:",
            "    pipeline.bbox_predictor = None",
            "    torch.cuda.empty_cache()",
            "    capture_memory('after_bbox_predictor_disabled')",
            "original_predict_bbox = pipeline.predict_bbox",
            "def guarded_predict_bbox(*args, **kwargs):",
            "    predicted = original_predict_bbox(*args, **kwargs)",
            "    before = int(predicted.shape[0]) if hasattr(predicted, 'shape') else None",
            "    guarded = predicted[: int(resource_limits['effective_max_parts'])]",
            "    after = int(guarded.shape[0]) if hasattr(guarded, 'shape') else None",
            "    write_resource_limits(device=str(device), resolved_torch_dtype=str(dtype).replace('torch.', ''), raw_aabb_count=before, effective_aabb_count=after, external_aabb=False)",
            "    return guarded",
            "pipeline.predict_bbox = guarded_predict_bbox",
            "generator = torch.Generator(device=device).manual_seed(seed)",
            "capture_memory('before_pipeline_call')",
            "with torch.inference_mode():",
            "    result = pipeline(",
            "        mesh_path=str(mesh_path),",
            "        aabb=aabb,",
            "        num_inference_steps=int(resource_limits['num_inference_steps']),",
            "        octree_resolution=int(resource_limits['octree_resolution']),",
            "        num_chunks=int(resource_limits['num_chunks']),",
            "        enable_pbar=False,",
            "        seed=seed,",
            "        generator=generator,",
            "        output_type='trimesh',",
            "    )",
            "capture_memory('after_pipeline_call')",
            "scene = result[0] if isinstance(result, (list, tuple)) else result",
            "if not isinstance(scene, trimesh.Scene):",
            "    raise RuntimeError(f'Expected trimesh.Scene from X-Part, got {type(scene).__name__}')",
            "scene.export(output_dir / 'x_part_primary.glb')",
            "capture_memory('after_primary_export')",
            "parts = []",
            "bbox_parts = []",
            "for index, (name, geom) in enumerate(scene.geometry.items()):",
            "    part_id = f'part-{index:03d}'",
            "    part_name = str(name) if name else part_id",
            "    part_path = parts_dir / f'{part_id}.glb'",
            "    trimesh.Scene([geom]).export(part_path)",
            "    bounds = geom.bounds.tolist() if getattr(geom, 'bounds', None) is not None else []",
            "    bbox = {'part_id': part_id, 'label': part_name, 'min': bounds[0] if len(bounds) == 2 else [], 'max': bounds[1] if len(bounds) == 2 else []}",
            "    parts.append({'part_id': part_id, 'label': part_name, 'path': str(part_path.relative_to(output_dir))})",
            "    bbox_parts.append(bbox)",
            "capture_memory('after_parts_export', exported_parts=len(parts))",
            "(output_dir / 'segmentation.json').write_text(json.dumps({'source': 'x-part', 'parts': parts}, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
            "(output_dir / 'bboxes.json').write_text(json.dumps({'source': 'x-part', 'parts': bbox_parts}, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
            "(output_dir / 'completion.json').write_text(json.dumps({'status': 'completed', 'adapter': 'x-part-upstream-subprocess', 'used_aabb': bool(aabb_path), 'resource_limits': json.loads(resource_limits_path.read_text(encoding='utf-8'))}, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
            "capture_memory('completion_written')",
            "torch.cuda.empty_cache()",
            "capture_memory('after_cuda_empty_cache')",
        ]
    )


def run_upstream_x_part(
    plan: ExecutionPlan,
    *,
    project_root: Path,
    managed_python: Path,
    model_root: Path | None,
    output_dir: Path,
    aabb_path: Path | None = None,
    runner: SubprocessRunner = subprocess.run,
) -> DecompositionArtifacts:
    if plan.params.export_format != "glb":
        raise SetupFailure(
            "The real upstream X-Part adapter currently supports GLB primary export only.",
            code="unsupported_runtime_export_format",
            details={"requested": plan.params.export_format, "supported": ["glb"]},
        )
    bundle_root = resolve_bundle_root(model_root)
    smoke = verify_x_part_import_smoke(project_root=project_root, managed_python=managed_python, model_root=model_root)
    if not smoke.get("ready"):
        missing_modules = smoke.get("missing_modules") or []
        raise SetupFailure(
            "X-Part adapter is fail-closed until its managed python, upstream pipeline, and Python dependencies are ready.",
            code="x_part_adapter_unavailable",
            details={**smoke, "missing_modules": missing_modules},
        )
    runtime_root = runtime_source_root(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = _resolve_x_part_timeout_seconds()
    min_available_gib = _resolve_x_part_min_available_memory_gib()
    resource_limits_path = output_dir / X_PART_RESOURCE_LIMITS_JSON_NAME
    adapter_resource_limits = resolve_adapter_x_part_resource_limits(plan.params, host_os_name=plan.context.host_facts.os_name)
    script = _build_subprocess_script(
        runtime_root=runtime_root,
        bundle_root=bundle_root,
        mesh_path=plan.mesh.mesh_path,
        output_dir=output_dir,
        params=plan.params,
        aabb_path=aabb_path,
        host_os_name=plan.context.host_facts.os_name,
    )
    command = subprocess_command(managed_python, "-c", script)
    subprocess_env = build_runtime_env(
        os.environ,
        pythonpath=(runtime_root, x_part_import_root(runtime_root / X_PART_PIPELINE_RELATIVE_PATH)),
        extra={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"},
    )
    memory_guard: dict[str, object] | None = None
    try:
        if runner is subprocess.run:
            result, memory_guard = _run_with_memory_guard(
                command,
                cwd=str(runtime_root / "XPart"),
                env=subprocess_env,
                timeout_seconds=timeout_seconds,
                min_available_gib=min_available_gib,
            )
        else:
            result = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(runtime_root / "XPart"),
                env=subprocess_env,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            args=command,
            returncode=-9,
            stdout=_normalize_timeout_output(exc.stdout),
            stderr=_normalize_timeout_output(exc.stderr) + f"\nX-Part subprocess timed out after {timeout_seconds} seconds.\n",
        )
        diagnostics = _persist_failure_diagnostics(
            output_dir=output_dir,
            bundle_root=bundle_root,
            mesh_path=plan.mesh.mesh_path,
            cwd=str(runtime_root / "XPart"),
            result=result,
            aabb_path=aabb_path,
            resource_limits_path=resource_limits_path,
            timeout_seconds=timeout_seconds,
            timed_out=True,
            memory_guard=memory_guard,
            adapter_resource_limits=adapter_resource_limits,
        )
        resource_limits_payload = _load_resource_limits_payload(resource_limits_path)
        resource_source = resource_limits_payload or adapter_resource_limits
        stdout_tail = _tail_text(result.stdout)
        stderr_tail = _tail_text(result.stderr)
        existing_artifacts = _existing_artifacts(output_dir)
        raise RuntimeFailure(
            _format_failure_message(
                reason="Upstream X-Part subprocess timed out",
                diagnostics=diagnostics,
                output_dir=output_dir,
                resource_limits_path=resource_limits_path,
                resource_summary=_resource_limits_summary(resource_source),
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                existing_artifacts=existing_artifacts,
                timeout_seconds=timeout_seconds,
                last_memory_stage=resource_source.get("last_memory_stage") if isinstance(resource_source, dict) else None,
                last_memory_event=_last_memory_event(resource_source if isinstance(resource_source, dict) else {}),
            ),
            code="x_part_subprocess_timeout",
            details={
                "timeout_seconds": timeout_seconds,
                "stdout": stdout_tail,
                "stderr": stderr_tail,
                "bundle_root": str(bundle_root),
                "mesh_path": str(plan.mesh.mesh_path),
                "output_dir": str(output_dir),
                "aabb_path": str(aabb_path) if aabb_path is not None else None,
                "diagnostics": diagnostics,
                "memory_guard": memory_guard,
                "resource_limits_path": str(resource_limits_path),
                "resource_limits_summary": _resource_limits_summary(resource_source),
                "existing_artifacts": existing_artifacts,
            },
        ) from exc
    if result.returncode != 0:
        memory_guard_triggered = bool(memory_guard and memory_guard.get("triggered"))
        diagnostics = _persist_failure_diagnostics(
            output_dir=output_dir,
            bundle_root=bundle_root,
            mesh_path=plan.mesh.mesh_path,
            cwd=str(runtime_root / "XPart"),
            result=result,
            aabb_path=aabb_path,
            resource_limits_path=resource_limits_path,
            timeout_seconds=timeout_seconds,
            memory_guard=memory_guard,
            adapter_resource_limits=adapter_resource_limits,
        )
        resource_limits_payload = _load_resource_limits_payload(resource_limits_path)
        resource_source = resource_limits_payload or adapter_resource_limits
        stdout_tail = _tail_text(result.stdout)
        stderr_tail = _tail_text(result.stderr)
        existing_artifacts = _existing_artifacts(output_dir)
        reason = "Upstream X-Part subprocess was stopped by the memory guard" if memory_guard_triggered else "Upstream X-Part subprocess failed"
        raise RuntimeFailure(
            _format_failure_message(
                reason=reason,
                diagnostics=diagnostics,
                output_dir=output_dir,
                resource_limits_path=resource_limits_path,
                resource_summary=_resource_limits_summary(resource_source),
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                existing_artifacts=existing_artifacts,
                returncode=result.returncode,
                last_memory_stage=resource_source.get("last_memory_stage") if isinstance(resource_source, dict) else None,
                last_memory_event=_last_memory_event(resource_source if isinstance(resource_source, dict) else {}),
            ),
            code="x_part_memory_guard_triggered" if memory_guard_triggered else "x_part_subprocess_failed",
            details={
                "returncode": result.returncode,
                "stdout": stdout_tail,
                "stderr": stderr_tail,
                "bundle_root": str(bundle_root),
                "mesh_path": str(plan.mesh.mesh_path),
                "output_dir": str(output_dir),
                "aabb_path": str(aabb_path) if aabb_path is not None else None,
                "diagnostics": diagnostics,
                "memory_guard": memory_guard,
                "resource_limits_path": str(resource_limits_path),
                "resource_limits_summary": _resource_limits_summary(resource_source),
                "existing_artifacts": existing_artifacts,
            },
        )
    return collect_artifacts(output_dir)
