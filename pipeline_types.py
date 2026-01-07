from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineFlags:
    generate: bool = False
    llm: bool = False
    compare: bool = False
    csv: bool = False
    stats: bool = False
    test: bool = False


@dataclass
class PipelineContext:
    config: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)


