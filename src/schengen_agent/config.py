"""Runtime configuration: switching providers does not change the agent loop."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    provider: str = "mock"
    model: str = "demo"
    max_steps: int = 6
    timeout_sec: float = 10.0
    data_dir: Path = ROOT / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("AGENT_PROVIDER", "mock").strip().lower()
        if provider not in {"mock", "openai", "gigachat"}:
            raise ValueError("AGENT_PROVIDER must be mock, openai or gigachat")
        default_model = {"mock": "demo", "openai": "gpt-4o-mini", "gigachat": "GigaChat-2"}
        max_steps = int(os.getenv("AGENT_MAX_STEPS", "6"))
        timeout_sec = float(os.getenv("AGENT_TIMEOUT_SEC", "10"))
        if not 1 <= max_steps <= 20:
            raise ValueError("AGENT_MAX_STEPS must be between 1 and 20")
        if timeout_sec <= 0:
            raise ValueError("AGENT_TIMEOUT_SEC must be positive")
        return cls(
            provider=provider,
            model=os.getenv("AGENT_MODEL", default_model[provider]),
            max_steps=max_steps,
            timeout_sec=timeout_sec,
            data_dir=Path(os.getenv("AGENT_DATA_DIR", str(ROOT / "data"))),
        )
