"""Boundary tests without network access or API keys."""
from types import SimpleNamespace

from schengen_agent.providers import OpenAIProvider, _parse_arguments


def test_malformed_model_generated_arguments_become_schema_error_sentinel():
    assert _parse_arguments('{invalid') == {"__invalid_json_arguments__": True}
    assert _parse_arguments('[1,2]') == {"__arguments_must_be_object__": True}
    assert _parse_arguments('{"country": "italy"}') == {"country": "italy"}


def test_openai_adapter_strips_gigachat_only_fields():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            tool_calls=[], content="OK"
        ))])

    provider = object.__new__(OpenAIProvider)
    provider.model = "test-model"
    provider.client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "get_visa_statistics", "arguments": '{"country":"france"}'
            }, "functions_state_id": "state-123"}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_visa_statistics", "content": "{}"},
    ]
    reply = provider.complete(messages, [])
    assert reply.content == "OK"
    assert "functions_state_id" not in seen["messages"][0]["tool_calls"][0]
    assert "name" not in seen["messages"][1]
    assert messages[0]["tool_calls"][0]["functions_state_id"] == "state-123"
