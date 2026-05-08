"""Typed failures for validation, setup, compatibility, and runtime boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HunyuanPartError(Exception):
    category: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "category": self.category,
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(HunyuanPartError):
    def __init__(self, message: str, *, code: str = "validation_error", details: dict[str, Any] | None = None) -> None:
        super().__init__("validation", code, message, details or {})


class CompatibilityFailure(HunyuanPartError):
    def __init__(self, message: str, *, code: str = "compatibility_failure", details: dict[str, Any] | None = None) -> None:
        super().__init__("compatibility_failure", code, message, details or {})


class DependencyFailure(HunyuanPartError):
    def __init__(self, message: str, *, code: str = "dependency_failure", details: dict[str, Any] | None = None) -> None:
        super().__init__("dependency_failure", code, message, details or {})


class SetupFailure(HunyuanPartError):
    def __init__(self, message: str, *, code: str = "setup_failure", details: dict[str, Any] | None = None) -> None:
        super().__init__("setup_failure", code, message, details or {})


class RuntimeFailure(HunyuanPartError):
    def __init__(self, message: str, *, code: str = "runtime_failure", details: dict[str, Any] | None = None) -> None:
        super().__init__("runtime_failure", code, message, details or {})
