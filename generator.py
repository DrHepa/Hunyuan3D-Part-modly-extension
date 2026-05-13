#!/usr/bin/env python3
"""Minimal Modly-loadable generator shell for Hunyuan3D-Part."""

from __future__ import annotations

import base64
import binascii
import hashlib
import sys
import os
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from services.generators.base import BaseGenerator
except ImportError:  # pragma: no cover - exercised in local tests only
    class BaseGenerator:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs


from runtime.config import evaluate_host_support, probe_torch_cuda_availability, resolve_host_facts, resolve_runtime_context
from runtime.config import normalize_params
from runtime.errors import RuntimeFailure, ValidationError
from runtime.platform_support import managed_python_path
from runtime.pipeline import run_pipeline_stage
from runtime.p3_sam import build_adapter_readiness, decompose_mesh, resolve_weight_path
from runtime.semantic_report import build_image_evidence
from runtime.validate import validate_mesh_path


class Hunyuan3DPartGenerator(BaseGenerator):
    """Managed-extension generator with a fail-closed P3-SAM adapter path."""

    _MESH_SUFFIXES = {".glb", ".obj", ".stl", ".ply"}
    _MESH_PATH_ALIASES = ("mesh_path", "mesh", "path", "file_path", "filePath", "local_path", "workspace_path")

    weight_owner_id = "p3sam"
    hf_repo = "tencent/Hunyuan3D-Part"
    download_check = "p3sam/p3sam.safetensors"

    def __init__(
        self,
        *args: Any,
        project_root: str | Path | None = None,
        runtime_context: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if getattr(self, "model_dir", None) is None and args:
            self.model_dir = args[0]
        if getattr(self, "outputs_dir", None) is None and len(args) > 1:
            self.outputs_dir = args[1]
        self.project_root = Path(project_root).resolve() if project_root else ROOT
        self.runtime_context = runtime_context or resolve_runtime_context(project_root=self.project_root)
        self._shell_loaded = False

    def _resolved_model_dir(self) -> Path | None:
        model_dir = getattr(self, "model_dir", None)
        if model_dir is None:
            return None
        return Path(model_dir)

    def _managed_venv_python(self) -> Path:
        return managed_python_path(self.project_root / "venv")

    def _setup_ready(self) -> bool:
        managed_python = self._managed_venv_python().resolve()
        current_python = Path(sys.executable).resolve()
        return managed_python.exists() or current_python == managed_python

    def _weight_path(self) -> Path:
        model_dir = self._resolved_model_dir()
        if model_dir is not None:
            return resolve_weight_path(model_dir)
        return resolve_weight_path(self.runtime_context.paths.model_root)

    def _effective_host_support(self) -> dict[str, object]:
        host_support = self.runtime_context.support.to_dict()
        failure = host_support.get("failure") if isinstance(host_support, dict) else None
        if not isinstance(failure, dict) or failure.get("code") != "cuda_not_visible":
            return host_support
        cuda_available = probe_torch_cuda_availability()
        if cuda_available is not True:
            return host_support

        dependency_map = dict(self.runtime_context.support.dependencies)
        refreshed_support = evaluate_host_support(
            resolve_host_facts(torch_cuda_probe=lambda: cuda_available),
            dependency_checker=lambda module: object() if dependency_map.get(module, False) else None,
        ).to_dict()
        refreshed_support["dependencies"] = dependency_map
        refreshed_support["source"] = "current_process_torch"
        return refreshed_support

    def _contract_machine_code(
        self,
        *,
        ok: bool,
        host_support: Mapping[str, object],
        setup: Mapping[str, object],
        runtime_adapter: Mapping[str, object],
        weights: Mapping[str, object],
    ) -> str:
        if ok:
            return "ready"
        if not bool(setup.get("ready", False)):
            return "setup_pending"
        if not bool(weights.get("ready", False)):
            return "setup_pending"
        if not bool(host_support.get("ready", False)):
            return "runtime_unavailable"
        if not bool(runtime_adapter.get("ready", False)):
            return "runtime_unavailable"
        return "runtime_unavailable"

    def _contract_reason(self, blockers: list[dict[str, object]]) -> str | None:
        if not blockers:
            return None
        blocker = blockers[0]
        if blocker.get("component") == "runtime_adapter":
            adapter_reason = self._runtime_adapter_blocker_reason(blocker)
            if adapter_reason:
                return adapter_reason
        failure = blocker.get("failure")
        if isinstance(failure, Mapping) and failure.get("message"):
            return str(failure["message"])
        if blocker.get("component") == "weights" and blocker.get("path"):
            return f"Missing required model weights: {blocker['path']}"
        if blocker.get("message"):
            return str(blocker["message"])
        if blocker.get("status"):
            return f"{blocker['component']} status={blocker['status']}"
        return f"{blocker['component']} is not ready"

    def _runtime_adapter_blocker_reason(self, blocker: Mapping[str, object]) -> str | None:
        components = blocker.get("components")
        if not isinstance(components, Mapping):
            return None
        import_smoke = components.get("import_smoke")
        if not isinstance(import_smoke, Mapping):
            return str(blocker.get("message")) if blocker.get("message") else None

        details: list[str] = []
        missing_modules = import_smoke.get("missing_modules")
        if isinstance(missing_modules, list) and missing_modules:
            details.append("missing_modules=" + ",".join(str(item) for item in missing_modules))
        native_blockers = import_smoke.get("native_blockers")
        if isinstance(native_blockers, list) and native_blockers:
            details.append("native_blockers=" + ",".join(str(item) for item in native_blockers))
        error_type = import_smoke.get("error_type")
        error = import_smoke.get("error")
        if error_type or error:
            details.append(f"error={error_type or 'Error'}: {error}".strip())
        stderr = import_smoke.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            details.append("stderr=" + stderr.strip().replace("\n", " | ")[:500])
        if details:
            return "Runtime adapter import smoke failed: " + "; ".join(details)
        message = blocker.get("message")
        if message:
            return str(message)
        status = import_smoke.get("status")
        if status:
            return f"Runtime adapter import smoke status={status}"
        return None

    def _runtime_unavailable_message(
        self,
        *,
        blockers: list[dict[str, object]],
        readiness: Mapping[str, object],
    ) -> str:
        components = ",".join(str(blocker.get("component", "unknown")) for blocker in blockers) or "unknown"
        reason = self._contract_reason(blockers) or "No detailed blocker reason was reported."
        details = readiness.get("details")
        if isinstance(details, Mapping):
            state = (
                f"setup_ready={bool(details.get('setup_ready', False))}; "
                f"weights_ready={bool(details.get('weights_ready', False))}; "
                f"adapter_ready={bool(details.get('adapter_ready', False))}; "
                f"host_support_ready={bool(details.get('host_support_ready', False))}; "
                f"inference_supported={bool(details.get('inference_supported', False))}"
            )
        else:
            state = "readiness details unavailable"
        return f"Runtime unavailable: blockers={components}; reason={reason}; {state}."

    def is_downloaded(self) -> bool:
        return self._weight_path().is_file()

    def readiness_status(self) -> dict[str, object]:
        host_support = self._effective_host_support()
        weights_ready = self.is_downloaded()
        model_dir = self._resolved_model_dir()
        setup_ready = self._setup_ready()
        adapter_readiness = build_adapter_readiness(
            project_root=self.project_root,
            managed_python=self._managed_venv_python(),
            model_root=model_dir if model_dir is not None else self.runtime_context.paths.model_root,
        )
        adapter_ready = bool(adapter_readiness.ready and setup_ready)
        setup_status = {
            "ready": setup_ready,
            "status": "prepared_shell",
            "owner": "electron",
            "prepared_shell": True,
            "message": (
                "Shell is prepared and the managed extension venv is available."
                if setup_ready
                else "Shell is prepared, but the managed extension venv is not available yet."
            ),
            "model_dir": str(model_dir) if model_dir is not None else None,
            "venv_python": str(self._managed_venv_python()),
        }
        weights_status = {
            "ready": weights_ready,
            "status": "downloaded" if weights_ready else "missing",
            "hf_repo": self.hf_repo,
            "download_check": self.download_check,
            "weight_owner_id": self.weight_owner_id,
            "model_dir": str(model_dir) if model_dir is not None else None,
            "path": str(self._weight_path()),
        }
        readiness = {
            "generator_class": self.__class__.__name__,
            "surface_owner": "electron",
            "headless_eligible": False,
            "shell_ready": True,
            "execution_ready": adapter_ready,
            "inference_ready": adapter_ready,
            "host_support": host_support,
            "platform_supported": bool(host_support.get("platform_supported", True)),
            "cuda_ready": bool(host_support.get("cuda_ready", False)),
            "native_wheels_ready": bool(host_support.get("native_wheels_ready", False)),
            "runtime_dependencies_ready": bool(host_support.get("runtime_dependencies_ready", False)),
            "inference_supported": bool(host_support.get("inference_supported", False)),
            "diagnostics": host_support.get("diagnostics", {}),
            "setup": setup_status,
            "runtime_adapter": adapter_readiness.to_dict(),
            "weights": weights_status,
        }
        blockers = self._collect_runtime_blockers(readiness)
        ok = bool(readiness["execution_ready"] and readiness["inference_ready"] and not blockers)
        machine_code = self._contract_machine_code(
            ok=ok,
            host_support=host_support,
            setup=setup_status,
            runtime_adapter=readiness["runtime_adapter"],
            weights=weights_status,
        )
        readiness.update(
            {
                "ok": ok,
                "machine_code": machine_code,
                "label_hint": "Ready" if ok else ("Setup pending" if machine_code == "setup_pending" else "Runtime unavailable"),
                "reason": self._contract_reason(blockers),
                "details": {
                    "blocker_count": len(blockers),
                    "host_support_ready": bool(host_support.get("ready", False)),
                    "inference_supported": bool(host_support.get("inference_supported", False)),
                    "setup_ready": setup_ready,
                    "adapter_ready": bool(readiness["runtime_adapter"].get("ready", False)),
                    "weights_ready": weights_ready,
                    "model_dir": str(model_dir) if model_dir is not None else None,
                    "venv_python": str(self._managed_venv_python()),
                },
                "blockers": blockers,
            }
        )
        return readiness

    def _collect_runtime_blockers(self, readiness: Mapping[str, object]) -> list[dict[str, object]]:
        blockers: list[dict[str, object]] = []
        setup = readiness["setup"]
        weights = readiness["weights"]
        runtime_adapter = readiness["runtime_adapter"]
        for key in ("setup", "weights"):
            section = readiness[key]
            if isinstance(section, Mapping) and not bool(section.get("ready", False)):
                if key == "setup" and section.get("status") == "prepared_shell":
                    continue
                blockers.append({"component": key, **dict(section)})
        host_support = readiness["host_support"]
        managed_runtime_ready = (
            isinstance(setup, Mapping)
            and bool(setup.get("ready", False))
            and isinstance(weights, Mapping)
            and bool(weights.get("ready", False))
            and isinstance(runtime_adapter, Mapping)
            and bool(runtime_adapter.get("ready", False))
        )
        if isinstance(host_support, Mapping) and not bool(host_support.get("ready", False)) and not managed_runtime_ready:
            blocker = {"component": "host_support", **dict(host_support)}
            blockers.append(blocker)
        for key in ("runtime_adapter",):
            section = readiness[key]
            if isinstance(section, Mapping) and not bool(section.get("ready", False)):
                blocker = {"component": key, **dict(section)}
                if blockers and blockers[-1].get("component") == "host_support":
                    blockers.insert(len(blockers) - 1, blocker)
                else:
                    blockers.append(blocker)
        return blockers

    def _emit_progress(
        self,
        progress_cb: Any | None,
        *,
        stage: str,
        percent: int,
        message: str,
    ) -> None:
        if progress_cb is None:
            return
        payload = {"stage": stage, "percent": percent, "message": message}
        try:
            progress_cb(percent, message)
        except TypeError:
            try:
                progress_cb(stage, percent, message)
            except TypeError:
                progress_cb(payload)

    def _normalize_generate_params(
        self,
        image_bytes: object,
        params: Mapping[str, object] | None,
        kwargs: Mapping[str, object],
    ) -> tuple[object, dict[str, object]]:
        resolved_image = image_bytes
        resolved_params: dict[str, object] = {}
        if isinstance(image_bytes, Mapping) and params is None:
            resolved_params.update(dict(image_bytes))
            resolved_image = None
        if params:
            resolved_params.update(dict(params))
        if kwargs:
            resolved_params.update(dict(kwargs))
        return resolved_image, resolved_params

    def _output_root(self) -> Path:
        return Path(getattr(self, "outputs_dir", None) or self.runtime_context.paths.artifacts_root)

    def _is_mesh_path_like(self, value: object) -> bool:
        if not isinstance(value, (str, Path)):
            return False
        return Path(value).suffix.lower() in self._MESH_SUFFIXES

    def _mesh_candidate_from_mapping(self, payload: Mapping[str, object]) -> object | None:
        for key in self._MESH_PATH_ALIASES:
            candidate = payload.get(key)
            if candidate is None:
                continue
            if isinstance(candidate, Mapping):
                nested = self._mesh_candidate_from_mapping(candidate)
                if nested is not None:
                    return nested
            elif isinstance(candidate, (str, Path, bytes, bytearray, memoryview)):
                return candidate
        return None

    def _decode_primary_mesh_base64(self, value: str) -> bytes | None:
        try:
            decoded = base64.b64decode(value.strip(), validate=True)
        except (binascii.Error, ValueError):
            return None
        return decoded if decoded.startswith(b"glTF") else None

    def _persist_primary_mesh_bytes(self, mesh_bytes: bytes, output_root: Path) -> Path:
        if not mesh_bytes.startswith(b"glTF"):
            raise ValidationError(
                "The required mesh input must be provided as the primary mesh input or params.mesh_path; byte/base64 payloads must be an identifiable GLB starting with the glTF magic.",
                code="missing_mesh_input",
                details={
                    "accepted_primary_inputs": [
                        "path string with .glb/.obj/.stl/.ply suffix",
                        "mapping containing mesh/path/file_path/filePath/local_path/workspace_path",
                        "GLB bytes beginning with glTF magic",
                        "base64-encoded GLB string beginning with glTF after decoding",
                    ],
                    "accepted_params": ["mesh_path", "mesh"],
                },
            )

        digest = hashlib.sha256(mesh_bytes).hexdigest()[:16]
        primary_inputs = output_root / ".primary-inputs"
        primary_inputs.mkdir(parents=True, exist_ok=True)
        mesh_path = primary_inputs / f"primary-mesh-{digest}.glb"
        if not mesh_path.exists() or mesh_path.read_bytes() != mesh_bytes:
            mesh_path.write_bytes(mesh_bytes)
        return mesh_path

    def _coerce_mesh_candidate(self, candidate: object, output_root: Path) -> object:
        if isinstance(candidate, (bytes, bytearray, memoryview)):
            return self._persist_primary_mesh_bytes(bytes(candidate), output_root)
        if isinstance(candidate, str) and not self._is_mesh_path_like(candidate):
            decoded_mesh = self._decode_primary_mesh_base64(candidate)
            if decoded_mesh is not None:
                return self._persist_primary_mesh_bytes(decoded_mesh, output_root)
        return candidate

    def _resolve_mesh_input(
        self,
        primary_input: object,
        params: Mapping[str, object],
        *,
        output_root: Path,
    ) -> tuple[object, object | None]:
        """Return ``(mesh_candidate, image_evidence_input)`` for legacy and mesh-primary calls."""
        legacy_mesh_path = params.get("mesh_path")
        if legacy_mesh_path is not None:
            return self._coerce_mesh_candidate(legacy_mesh_path, output_root), primary_input

        params_mesh = self._mesh_candidate_from_mapping(params)
        if params_mesh is not None:
            return self._coerce_mesh_candidate(params_mesh, output_root), primary_input

        if self._is_mesh_path_like(primary_input):
            return primary_input, None

        if isinstance(primary_input, Mapping):
            candidate = self._mesh_candidate_from_mapping(primary_input)
            if candidate is not None:
                return self._coerce_mesh_candidate(candidate, output_root), None

        if isinstance(primary_input, str):
            decoded_mesh = self._decode_primary_mesh_base64(primary_input)
            if decoded_mesh is not None:
                return self._persist_primary_mesh_bytes(decoded_mesh, output_root), None

        if isinstance(primary_input, (bytes, bytearray, memoryview)):
            return self._persist_primary_mesh_bytes(bytes(primary_input), output_root), None

        raise ValidationError(
            "The required mesh input must be provided as the primary mesh input or params.mesh_path.",
            code="missing_mesh_input",
            details={
                "accepted_primary_inputs": [
                    "path string with .glb/.obj/.stl/.ply suffix",
                    "mapping containing mesh/path/file_path/filePath/local_path/workspace_path",
                    "GLB bytes beginning with glTF magic",
                    "base64-encoded GLB string beginning with glTF after decoding",
                ],
                "accepted_params": ["mesh_path", "mesh"],
            },
        )

    def load(self) -> "Hunyuan3DPartGenerator":
        self._shell_loaded = True
        return self

    def _mesh_search_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        def add_root(candidate: object) -> None:
            if candidate is None:
                return
            path = Path(candidate)
            if path in seen:
                return
            seen.add(path)
            roots.append(path)

        outputs_dir = getattr(self, "outputs_dir", None)
        if outputs_dir is not None:
            output_path = Path(outputs_dir)
            add_root(output_path)
            add_root(output_path.parent)

        workspace_dir = os.environ.get("WORKSPACE_DIR")
        if workspace_dir:
            add_root(workspace_dir)

        return roots

    def generate(
        self,
        image_bytes: object = None,
        params: Mapping[str, object] | None = None,
        progress_cb: Any | None = None,
        cancel_event: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, object]:
        self.load()
        self._emit_progress(
            progress_cb,
            stage="shell",
            percent=5,
            message="Generator shell prepared; validating mesh input and runtime readiness.",
        )

        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise RuntimeFailure(
                "Generation cancelled before runtime validation completed.",
                code="generation_cancelled",
            )

        resolved_image, resolved_params = self._normalize_generate_params(image_bytes, params, kwargs)
        output_root = self._output_root()
        mesh_path, image_evidence_input = self._resolve_mesh_input(
            resolved_image,
            resolved_params,
            output_root=output_root,
        )

        validated_mesh = validate_mesh_path(mesh_path, search_roots=self._mesh_search_roots())
        normalized_params = normalize_params(resolved_params)
        image_evidence = build_image_evidence(image_evidence_input)
        readiness = self.readiness_status()
        blockers = self._collect_runtime_blockers(readiness)

        self._emit_progress(
            progress_cb,
            stage="readiness",
            percent=15,
            message="Runtime shell validated inputs and collected execution blockers.",
        )

        if blockers:
            raise RuntimeFailure(
                self._runtime_unavailable_message(blockers=blockers, readiness=readiness),
                code="runtime_unavailable",
                details={
                    "blockers": blockers,
                    "host_support": readiness["host_support"],
                    "mesh_path": str(validated_mesh.mesh_path),
                    "mesh_format": validated_mesh.mesh_format,
                    "params": normalized_params.to_dict(),
                    "image_input_present": image_evidence_input is not None,
                    "shell_loaded": self._shell_loaded,
                },
            )
        result = decompose_mesh(
            {"mesh": validated_mesh.mesh_path},
            resolved_params,
            output_dir=output_root,
            image_evidence=image_evidence,
            runtime_context=self.runtime_context,
            project_root=self.project_root,
            runtime_adapter=lambda plan: run_pipeline_stage(
                plan,
                project_root=self.project_root,
                managed_python=self._managed_venv_python(),
                model_root=self._resolved_model_dir() or self.runtime_context.paths.model_root,
                output_dir=output_root,
            ),
        )
        if isinstance(result, Mapping) and result.get("primary_mesh"):
            return str(result["primary_mesh"])
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        return self.generate(*args, **kwargs)
