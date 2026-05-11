"""Pure platform compatibility helpers for managed Modly runtimes."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence


NATIVE_DEPENDENCY_MODULES = (
    "torch",
    "spconv",
    "spconv.pytorch",
    "cumm",
    "cumm.core_cc",
    "torch_scatter",
    "torch_cluster",
)


def platform_system(value: str | None = None) -> str:
    """Return a normalized platform.system() value."""

    return (value or platform.system() or "").strip().lower()


def is_windows(system: str | None = None) -> bool:
    return platform_system(system) == "windows"


def _validate_configured_executable(path: str | Path, *, system: str | None = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Configured executable does not exist: {resolved}")
    if not is_windows(system) and not os.access(resolved, os.X_OK):
        raise PermissionError(f"Configured executable is not executable: {resolved}")
    return resolved


def managed_python_path(venv_dir: Path, platform_system: str | None = None, configured: str | Path | None = None) -> Path:
    if configured is not None:
        return _validate_configured_executable(configured, system=platform_system)
    if is_windows(platform_system):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def managed_pip_path(venv_dir: Path, platform_system: str | None = None, configured: str | Path | None = None) -> Path:
    if configured is not None:
        return _validate_configured_executable(configured, system=platform_system)
    if is_windows(platform_system):
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def managed_bin_dir(venv_dir: Path, platform_system: str | None = None) -> Path:
    return venv_dir / ("Scripts" if is_windows(platform_system) else "bin")


def _compose_path_entries(entries: Sequence[str | Path], current: str | None = None) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for entry in [*(str(item) for item in entries), *(str(current or "").split(os.pathsep))]:
        normalized = entry.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return os.pathsep.join(values)


def build_runtime_env(
    base: Mapping[str, str] | None = None,
    *,
    prepend_path: Sequence[str | Path] = (),
    pythonpath: Sequence[str | Path] = (),
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Compose subprocess env with platform-native path separators."""

    env = dict(base or os.environ)
    if prepend_path:
        env["PATH"] = _compose_path_entries(prepend_path, env.get("PATH"))
    if pythonpath:
        env["PYTHONPATH"] = _compose_path_entries(pythonpath, env.get("PYTHONPATH"))
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def subprocess_command(*parts: str | Path) -> list[str]:
    """Return a list command; never shell-quote paths with spaces."""

    return [str(part) for part in parts]


def cleanup_with_retries(path: Path, *, attempts: int = 3, delay_seconds: float = 0.05) -> dict[str, object]:
    """Best-effort cleanup that tolerates transient Windows file locks."""

    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return {"removed": True, "attempts": attempt, "path": str(path)}
        except FileNotFoundError:
            return {"removed": False, "attempts": attempt, "path": str(path), "status": "missing"}
        except PermissionError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_seconds * attempt)
                continue
            return {
                "removed": False,
                "attempts": attempt,
                "path": str(path),
                "status": "locked",
                "error": str(exc),
            }
        except OSError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_seconds * attempt)
                continue
            break
    return {
        "removed": False,
        "attempts": attempts,
        "path": str(path),
        "status": "failed",
        "error": str(last_error) if last_error else None,
    }


def _cuda_candidates(system: str, env: Mapping[str, str]) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"):
        value = env.get(name)
        if value:
            candidates.append((Path(value), name))
    if is_windows(system):
        candidates.extend(
            (Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{version}"), "windows-default")
            for version in ("13.0", "12.8", "12.6", "12.4", "12.1", "11.8")
        )
    else:
        candidates.extend(
            (Path(value), "linux-default")
            for value in ("/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12.4")
        )
    deduped: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, source in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((path, source))
    return deduped


def detect_cuda_toolkit(platform_system: str | None = None, env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Detect CUDA toolkit layout without mutating runtime state."""

    system = globals()["platform_system"](platform_system)
    current_env = env or os.environ
    inspected: list[str] = []
    nvcc_name = "nvcc.exe" if is_windows(system) else "nvcc"
    lib_candidates = (Path("lib") / "x64",) if is_windows(system) else (Path("lib64"), Path("targets") / "sbsa-linux" / "lib", Path("lib"))
    for root, source in _cuda_candidates(system, current_env):
        nvcc_path = root / "bin" / nvcc_name
        include = root / "include" / "cuda.h"
        lib_dirs = [root / relative for relative in lib_candidates if (root / relative).exists()]
        inspected.append(str(root))
        if nvcc_path.exists() and include.exists() and lib_dirs:
            return {
                "ready": True,
                "status": "ready",
                "source": source,
                "toolkit_root": str(root),
                "nvcc_path": str(nvcc_path),
                "lib_dirs": [str(item) for item in lib_dirs],
                "reason": "CUDA toolkit layout is available.",
                "next_action": "None. CUDA toolkit detection passed.",
            }
    return {
        "ready": False,
        "status": "blocked",
        "source": None,
        "toolkit_root": None,
        "nvcc_path": None,
        "lib_dirs": [],
        "reason": f"No CUDA toolkit with {nvcc_name}, include/cuda.h, and libraries was found. Inspected: {', '.join(inspected) or 'none'}.",
        "next_action": "Install CUDA toolkit or set CUDA_PATH/CUDA_HOME to a valid toolkit root before enabling inference.",
    }


def probe_native_dependencies(python: Path, modules: Sequence[str] = NATIVE_DEPENDENCY_MODULES) -> dict[str, object]:
    """Probe native dependency importability in managed Python without inference."""

    if not python.exists():
        return {
            "ready": False,
            "status": "pending",
            "managed_python": str(python),
            "imports": {},
            "missing_modules": [],
            "reason": "Managed Python is missing; native dependencies cannot be probed yet.",
            "next_action": "Run managed setup before probing native runtime dependencies.",
        }
    script = "\n".join(
        [
            "import importlib, json",
            f"modules = {list(modules)!r}",
            "payload = {}",
            "for name in modules:",
            "    try:",
            "        if name == 'torch':",
            "            importlib.import_module('torch')  # import first so Windows DLL paths are initialized",
            "        else:",
            "            importlib.import_module(name)",
            "        payload[name] = {'ready': True, 'status': 'ready'}",
            "    except Exception as exc:",
            "        payload[name] = {'ready': False, 'status': 'missing', 'error_type': type(exc).__name__, 'error': str(exc)}",
            "print(json.dumps(payload, sort_keys=True))",
        ]
    )
    try:
        result = subprocess.run([str(python), "-c", script], check=False, capture_output=True, text=True)
    except Exception as exc:
        return {
            "ready": False,
            "status": "blocked",
            "managed_python": str(python),
            "imports": {},
            "missing_modules": list(modules),
            "reason": f"Native dependency probe could not start: {exc}",
            "next_action": "Repair the managed Python executable before probing native dependencies.",
            "error_type": type(exc).__name__,
        }
    try:
        imports = json.loads(result.stdout.strip() or "{}") if result.returncode == 0 else {}
    except json.JSONDecodeError:
        imports = {}
    missing = [name for name in modules if not imports.get(name, {}).get("ready", False)]
    return {
        "ready": not missing,
        "status": "ready" if not missing else "blocked",
        "managed_python": str(python),
        "imports": imports,
        "missing_modules": missing,
        "reason": "Native dependency imports are ready." if not missing else "Native dependency imports are missing or blocked.",
        "next_action": "None. Native dependency probe passed." if not missing else "Install/repair the reported native dependencies before enabling inference.",
        "returncode": result.returncode,
        "stderr_tail": (result.stderr or "")[-4000:],
    }


def memory_guard_status(platform_system: str | None = None, *, proc_meminfo: Path = Path("/proc/meminfo")) -> dict[str, object]:
    system = globals()["platform_system"](platform_system)
    if not is_windows(system) and proc_meminfo.exists():
        return {"guard_status": "enforced", "enforced": True, "source": str(proc_meminfo)}
    try:
        import psutil  # type: ignore

        available_gib = psutil.virtual_memory().available / (1024**3)
        return {
            "guard_status": "fallback",
            "enforced": True,
            "source": "psutil.virtual_memory",
            "available_gib": round(available_gib, 3),
        }
    except Exception as exc:
        return {
            "guard_status": "degraded",
            "enforced": False,
            "source": None,
            "reason": f"/proc memory data unavailable and psutil fallback failed: {exc}",
            "next_action": "Install psutil for memory diagnostics on non-/proc platforms or rely on external process limits.",
        }
