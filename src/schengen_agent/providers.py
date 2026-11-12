"""LLM provider adapters. The agent loop is provider-independent."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelReply:
    content: str | None = None
    calls: list[ToolCall] | None = None


class Provider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply: ...


class MockProvider:
    """Deterministic offline demonstration, not a real language model.

    It selects appropriate tool calls using simple keywords, allowing classroom
    testing without API access. Real providers replace this class unchanged.
    """

    def __init__(self):
        self.planned: list[tuple[str, dict]] | None = None
        self.index = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        user_text = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )
        if self.planned is None:
            lower = user_text.lower()
            country = next((
                slug for slug, names in {
                    "france": ["франц", "france", "француз"],
                    "italy": ["итал", "italy"],
                    "spain": ["испан", "spain"],
                }.items() if any(n in lower for n in names)
            ), None)
            if country is None:
                return ModelReply(content=(
                    "Я демонстрационный визовый агент. Назовите Францию, "
                    "Италию или Испанию, и я покажу учебные данные и ссылки "
                    "на официальные порталы."
                ))
            city = next((
                slug for slug, roots in {
                    "moscow": ["моск", "moscow"],
                    "saint_petersburg": ["петербург", "питер", "saint_petersburg"],
                    "ekaterinburg": ["екатеринбург", "ekaterinburg"],
                }.items() if any(root in lower for root in roots)
            ), None)
            base: dict[str, Any] = {"country": country}
            if city:
                base["city"] = city
            only_cases = any(t in lower for t in ["кейс", "случа", "похож"])
            only_docs = any(t in lower for t in ["документ", "требован", "список"])
            only_stats = any(t in lower for t in ["статист", "сколько ждать", "срок", "очеред", "индекс"])
            explicit_all = sum((only_cases, only_docs, only_stats)) >= 2 or (
                not any((only_cases, only_docs, only_stats))
            )
            self.planned = []
            if explicit_all or only_docs:
                self.planned.append(("get_visa_requirements", base.copy()))
            if explicit_all or only_stats:
                self.planned.append(("get_visa_statistics", base.copy()))
            if explicit_all or only_cases:
                if "раньше" in lower or "был шенген" in lower:
                    base = dict(base, had_schengen_before=True)
                self.planned.append(("search_visa_cases", base.copy()))
        if self.index < len(self.planned):
            name, args = self.planned[self.index]
            self.index += 1
            return ModelReply(calls=[ToolCall(f"mock-{self.index}", name, args)])
        observations = [
            m.get("content", "") for m in messages if m["role"] == "tool"
        ]
        successes, errors = [], []
        for text in observations:
            try:
                obj = json.loads(text)
                if "error" in obj:
                    errors.append(obj["error"]["message"])
                elif "case_count" in obj:
                    successes.append(
                        f"В учебной выборке {obj['case_count']} кейсов; "
                        f"медиана ожидания записи: {obj['median_appointment_wait_days']} дн., "
                        f"рассмотрения: {obj['median_processing_days']} дн. "
                        f"Доля положительных решений: {obj['observed_approval_rate']}."
                    )
                elif "official_portal" in obj:
                    successes.append(
                        "Официальный визовый портал: " + obj["official_portal"]
                        + " (актуальные требования необходимо проверить на сайте)."
                    )
                elif "cases" in obj:
                    successes.append(f"Найдено {len(obj['cases'])} вымышленных похожих кейсов.")
            except (ValueError, TypeError, KeyError):
                errors.append("Не удалось разобрать ответ инструмента.")
        out = ["ДЕМОНСТРАЦИЯ: все приведённые кейсы и показатели синтетические."]
        out.extend(successes)
        if errors:
            out.append("Не удалось получить часть сведений: " + "; ".join(errors))
        if not successes:
            out.append("Проверенные сведения недоступны; не делайте выводов о реальной визовой практике.")
        return ModelReply(content="\n".join(out))


class OpenAIProvider:
    def __init__(self, model: str, timeout_sec: float):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=timeout_sec,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, json.loads(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        return ModelReply(content=msg.content, calls=calls)


class GigaChatProvider:
    """Adapter for the GigaChat Python SDK's function-calling interface."""

    def __init__(self, model: str, timeout_sec: float):
        from gigachat import GigaChat

        credentials = os.environ["GIGACHAT_CREDENTIALS"]
        # Never turn TLS verification off just to make the classroom demo work.
        self.client = GigaChat(
            credentials=credentials,
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=model,
            timeout=timeout_sec,
            verify_ssl_certs=True,
            ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        from gigachat.models import Chat, Messages, MessagesRole

        converted = []
        for m in messages:
            role = m["role"]
            if role == "tool":
                converted.append(Messages(
                    role=MessagesRole.FUNCTION,
                    name=m.get("name", "tool"),
                    content=m["content"],
                ))
            elif role == "assistant" and m.get("tool_calls"):
                # GigaChat uses one function_call per message (not OpenAI tool_calls).
                for tc in m["tool_calls"]:
                    converted.append(Messages(
                        role=MessagesRole.ASSISTANT,
                        content="",
                        function_call={
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        },
                    ))
            else:
                converted.append(Messages(role=role, content=m.get("content") or ""))
        available = [
            {
                "name": spec["function"]["name"],
                "description": spec["function"]["description"],
                "parameters": spec["function"]["parameters"],
            }
            for spec in tools
        ]
        result = self.client.chat(Chat(messages=converted, functions=available, function_call="auto"))
        message = result.choices[0].message
        if message.function_call:
            call = message.function_call
            args = call.arguments
            if isinstance(args, str):
                args = json.loads(args)
            return ModelReply(
                calls=[ToolCall(f"gigachat-{len(messages)}", call.name, args)]
            )
        return ModelReply(content=message.content)


def make_provider(provider: str, model: str, timeout_sec: float) -> Provider:
    if provider == "mock":
        return MockProvider()
    if provider == "openai":
        return OpenAIProvider(model, timeout_sec)
    if provider == "gigachat":
        return GigaChatProvider(model, timeout_sec)
    raise ValueError(f"Unknown provider: {provider}")
