from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

LogFn = Callable[[str], None]


@dataclass
class Support:
    ok: bool
    reason: str = ""


@dataclass
class EnginePayload:
    adapter: str
    env_name: str
    command: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    ignored_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "env_name": self.env_name,
            "command": self.command,
            "ignored_fields": self.ignored_fields,
            "notes": self.notes,
            "extra": self.extra,
        }


@dataclass
class EvalResult:
    success_rate: float
    mean_return: float
    episodes: int
    video_path: Path | None
    passed: bool
    notes: list[str] = field(default_factory=list)


class Adapter(Protocol):
    name: str

    def support(self, spec: dict[str, Any]) -> Support:
        ...

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        ...

    def launch(
        self,
        spec: dict[str, Any],
        payload: EnginePayload,
        run_dir: Path,
        log: LogFn,
    ) -> EvalResult:
        ...
