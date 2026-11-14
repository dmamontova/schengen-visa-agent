"""Tool execution boundary: whitelist, argument validation, bounded retries and logging."""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import Any

from pydantic import ValidationError

from .data import DataUnavailable, LocalData
from .schemas import TOOL_MODELS

LOGGER = logging.getLogger(__name__)


class ToolTimeout(Exception):
    pass


class ToolServiceUnavailable(Exception):
    """A transient service failure, as opposed to a local file I/O error."""


class FaultInjector:
    """Deterministically inject failures before a real tool operation for the demo."""

    def __init__(self, faults: dict[str, list[str]] | None = None):
        self.queues = {name: deque(items) for name, items in (faults or {}).items()}

    def before_call(self, name: str):
        q = self.queues.get(name)
        if q:
            kind = q.popleft()
            if kind == "timeout":
                raise ToolTimeout("Учебная имитация таймаута")
            if kind == "unavailable":
                raise ToolServiceUnavailable("Учебная имитация недоступного сервиса")


def _error(kind: str, message: str, hint: str) -> dict:
    return {"error": {"type": kind, "message": message, "hint": hint}}


class ToolExecutor:
    def __init__(self, data: LocalData, faults: FaultInjector | None = None):
        self.data = data
        self.faults = faults or FaultInjector()
        self.call_log: list[dict[str, Any]] = []
        # Non-persistent, trusted results for grounding numerical final answers.
        # These are intentionally NOT included in the public sanitized trace.
        self.successful_results: dict[str, dict] = {}

    def execute(self, name: str, arguments: dict) -> dict:
        start = time.monotonic()
        # Never reuse an older success if the latest call fails.
        self.successful_results.pop(name, None)
        if name not in TOOL_MODELS:
            response = _error("unknown_tool", "Инструмент недоступен.", "Выбери разрешённый инструмент.")
            self._record(name, response, start)
            return response

        try:
            args = TOOL_MODELS[name].model_validate(arguments)
        except ValidationError as exc:
            details = [
                {"field": ".".join(map(str, issue["loc"])), "message": issue["msg"]}
                for issue in exc.errors()[:3]
            ]
            response = _error(
                "validation", "Недопустимые аргументы инструмента.",
                "Исправь аргументы по JSON-схеме: " + json.dumps(details, ensure_ascii=False),
            )
            self._record(name, response, start)
            return response

        # Safe read-only tools may be retried once. No retries for validation/no data.
        for attempt in range(2):
            try:
                self.faults.before_call(name)
                result = getattr(self.data, name)(args)
                self.successful_results[name] = result
                self._record(name, result, start, attempts=attempt + 1)
                return result
            except DataUnavailable as exc:
                response = _error("no_data", str(exc), "Измени фильтры или честно сообщи, что данных нет.")
                self._record(name, response, start, attempts=attempt + 1)
                return response
            except (ToolTimeout, ToolServiceUnavailable) as exc:
                if attempt == 0:
                    continue
                response = _error("timeout", "Сервис данных временно не ответил.",
                                  "Не выдумывай результат; сообщи об ограничении.")
                LOGGER.warning("Tool %s failed after 2 attempts: %s", name, type(exc).__name__)
                self._record(name, response, start, attempts=attempt + 1)
                return response
            except OSError as exc:
                LOGGER.warning("Local file access failed for %s: %s", name, type(exc).__name__)
                response = _error(
                    "data_error", "Не удалось прочитать локальный файл данных.",
                    "Проверьте наличие и доступность JSON-файлов данных.",
                )
                self._record(name, response, start, attempts=attempt + 1)
                return response
            except (ValueError, KeyError, TypeError) as exc:
                LOGGER.exception("Invalid local data for %s", name)
                response = _error("data_error", "Ошибка формата локальных данных.",
                                  "Не используй ненадёжный результат.")
                self._record(name, response, start, attempts=attempt + 1)
                return response
        raise AssertionError("Unreachable")

    def _record(self, name: str, result: dict, start: float, attempts: int = 0):
        # No user prompt, API keys or raw case text in the trace.
        entry = {
            "tool": name,
            "duration_ms": round(1000 * (time.monotonic() - start), 2),
            "attempts": attempts,
            "error_type": result.get("error", {}).get("type"),
            "result_items": len(result.get("cases", [])) if "cases" in result else None,
        }
        self.call_log.append(entry)
